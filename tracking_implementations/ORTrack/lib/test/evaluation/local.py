from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    settings.davis_dir = ''
    settings.got10k_path = '/home/furkan/Desktop/CS/altek/tracking_implementations/ORTrack/data/got10k'
    settings.lasot_path = '/home/furkan/Desktop/CS/altek/tracking_implementations/ORTrack/data/lasot'
    settings.network_path = '/home/furkan/Desktop/CS/altek/tracking_implementations/ORTrack/pretrained_models'
    settings.nfs_path = '/home/furkan/Desktop/CS/altek/tracking_implementations/ORTrack/data/nfs'
    settings.otb_path = '/home/furkan/Desktop/CS/altek/tracking_implementations/ORTrack/data/otb'
    settings.result_plot_path = '/home/furkan/Desktop/CS/altek/tracking_implementations/ORTrack/output/test/result_plots'
    settings.results_path = '/home/furkan/Desktop/CS/altek/tracking_implementations/ORTrack/output/test/tracking_results'
    settings.segmentation_path = '/home/furkan/Desktop/CS/altek/tracking_implementations/ORTrack/output/test/segmentation_results'
    settings.trackingnet_path = '/home/furkan/Desktop/CS/altek/tracking_implementations/ORTrack/data/trackingnet'
    settings.uav_path = '/home/furkan/Desktop/CS/altek/tracking_implementations/ORTrack/data/uav'
    settings.vot_path = '/home/furkan/Desktop/CS/altek/tracking_implementations/ORTrack/data/VOT2019'
    return settings
