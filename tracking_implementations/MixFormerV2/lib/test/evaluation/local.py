from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    settings.prj_dir = '/home/furkan/Desktop/CS/altek/tracking_implementations/MixFormerV2'
    settings.save_dir = '/home/furkan/Desktop/CS/altek/tracking_implementations/MixFormerV2/output'
    settings.results_path = '/home/furkan/Desktop/CS/altek/tracking_implementations/MixFormerV2/output/test/results'
    settings.segmentation_path = '/home/furkan/Desktop/CS/altek/tracking_implementations/MixFormerV2/output/test/segmentation_results'
    return settings
