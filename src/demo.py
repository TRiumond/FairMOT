from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
from opts import opts
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import TrackSqlSaver

logger.setLevel(logging.INFO)


def demo(opt):
    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    frame_rate = dataloader.frame_rate
    saver = TrackSqlSaver(db_name=opt.database,
                          table_name=opt.table_name,
                          tracking_session_id=opt.session_id,
                          opt=opt,
                          dataloader=dataloader,
                          data_type="mot",
                          frame_rate=frame_rate,
                          use_cuda=opt.gpus != [-1],
                          save_video_path=opt.output_file)
    saver.eval(show_image=opt.show_image)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
