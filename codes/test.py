import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    util.mkdirs(
        (path for key, path in opt['path'].items()
         if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    #### Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test audio files in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    model = create_model(opt)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        test_start_time = time.time()
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['psnr_y'] = []

        for data in test_loader:
            need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
            model.feed_data(data, need_GT=need_GT)
            audio_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
            audio_name = osp.splitext(osp.basename(audio_path))[0]

            model.test()
            audio_samples = model.get_current_audio_samples(need_GT=need_GT)

            sr_audio = audio_samples['SR']

            suffix = opt['suffix']
            if suffix:
                save_audio_path = osp.join(dataset_dir, audio_name + suffix + '.wav')
            else:
                save_audio_path = osp.join(dataset_dir, audio_name + '.wav')
            util.save_audio(sr_audio, save_audio_path)
            logger.info(audio_name)

if __name__ == '__main__':
    main()
