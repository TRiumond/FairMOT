from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import motmetrics as mm
import numpy as np
import torch
import pandas as pd

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts
import sqlite3
import json
from json import JSONEncoder
import numpy


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        types = ['float32', 'float64']
        if obj.dtype.name in types:
            obj = obj.astype(np.float64)
        if isinstance(obj, numpy.ndarray):
            if obj.dtype.name in types:
                obj = obj.astype(np.float64)

            return obj.tolist()

        return JSONEncoder.default(self, obj)


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


class TrackSaver(object):
    def __init__(self, opt, dataloader, data_type, frame_rate=30, use_cuda=True, save_video_path=None):
        self.opt = opt
        self.dataloader = dataloader
        self.data_type = data_type
        self.use_cuda = use_cuda
        self.frame_rate = frame_rate
        self.video_saver = None
        if save_video_path is not None:
            self.set_video_saver(save_video_path)

    def set_video_saver(self, path):
        self.video_saver = cv2.VideoWriter(path, -1, self.dataloader.frame_rate,
                                           (self.dataloader.vw, self.dataloader.vh))

    def send_result(self, result, raw_img):
        pass

    def send_image(self, img0, online_tlwhs, online_ids, frame_id, fps):
        online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id, fps=fps)
        if self.video_saver is not None:
            self.video_saver.write(online_im)
        return online_im

    def eval(self, skip_frame=1, show_image=False):
        tracker = JDETracker(self.opt, frame_rate=self.frame_rate)
        timer = Timer()
        frame_id = 0

        for i, (path, img, img0) in enumerate(self.dataloader):
            if i % skip_frame != 0:
                continue
            if frame_id % 20 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

            # run tracking
            timer.tic()
            if self.use_cuda:
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
            else:
                blob = torch.from_numpy(img).unsqueeze(0)
            online_targets = tracker.update(blob, img0)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.opt.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            timer.toc()
            tmp_result = {"frame_id": frame_id + 1, "bounding_box": online_tlwhs, "ids": online_ids,
                          "scores": online_scores}
            self.send_result(tmp_result, raw_img=img0)

            frame_id += 1
            if show_image:
                online_im = self.send_image(img0, online_tlwhs, online_ids, frame_id,
                                            1. / max(1e-5, timer.average_time))
                cv2.imshow('Result', online_im)
        if self.video_saver is not None:
            self.video_saver.release()
        return frame_id, timer.average_time, timer.calls


class TrackSqlSaver(TrackSaver):
    def __init__(self, db_name: str, table_name: str, tracking_session_id: str, opt, dataloader, data_type: str,
                 frame_rate: int = 30, use_cuda: bool = True, save_video_path=None):
        TrackSaver.__init__(self, opt, dataloader, data_type, frame_rate, use_cuda, save_video_path)
        self.db_name = db_name
        self.table_name = table_name
        self.db = sqlite3.connect(self.db_name)

        self.tracking_session_id = tracking_session_id
        self.cur = self.db.cursor()
        self.schema = ["tracking_session_id",
                       "frame_id",
                       "bounding_box",
                       "ids",
                       "scores",
                       "img"]
        sql_create_table = """
        CREATE TABLE IF NOT EXISTS %s (tracking_session_id string,
                         frame_id integer,
                         bounding_box string,
                         ids string,
                         scores string, 
                         img blob)
        """ % self.table_name

        self.cur.execute(sql_create_table)
        self.db.commit()

    def send_result(self, result, raw_img):
        _, enc = cv2.imencode(".png", raw_img)
        frame_id = result['frame_id']
        bounding_box = json.dumps(result['bounding_box'], cls=NumpyArrayEncoder)
        ids = json.dumps(result['ids'], cls=NumpyArrayEncoder)
        scores = str(result['scores'])
        self.cur.execute("insert into %s values(?,?,?,?,?,?)" % self.table_name, (self.tracking_session_id,
                                                                                  frame_id,
                                                                                  bounding_box,
                                                                                  ids,
                                                                                  scores,
                                                                                  enc))
        self.db.commit()


class TackPandasSaver(TrackSaver):
    def __init__(self, tracking_session_id: str, opt, dataloader, data_type: str,
                 frame_rate: int = 30, use_cuda: bool = True, save_video_path=None, output_folder: str = None):
        TrackSaver.__init__(self, opt, dataloader, data_type, frame_rate, use_cuda, save_video_path)
        self.track_session_id = tracking_session_id
        self.output_folder = output_folder

    def send_result(self, result: dict, raw_img):
        frame_id = result['frame_id']
        result_data = []
        for bounding_box, id, score in zip(result['bounding_box'], result['ids'], result['scores']):
            x, y, w, h = bounding_box
            tmp = {"x": x, 'y': y, 'w': w, 'h': h, 'id': id, 'score': score, 'frame_id': frame_id}
            result_data.append(tmp)
        path = "%s-%s.csv" % (self.track_session_id, frame_id)
        if self.output_folder is not None:
            path = os.path.join(self.output_folder, path)

        pd.DataFrame(result_data).to_csv(path)


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    # for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        # if i % 8 != 0:
        # continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        # online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                # online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        # results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    # write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        # seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        # seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        # seqs_str = '''MOT16-06 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOT17_test_public_dla34',
         show_image=False,
         save_images=False,
         save_videos=False)
