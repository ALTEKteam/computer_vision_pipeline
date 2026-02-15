import os

# Mevcut çalışma dizinini al (Örn: /home/furkan/Desktop/.../ORTrack)
base_path = os.getcwd()
print(f"ORTrack Ana Dizini Algılandı: {base_path}")

# ----------------------------------------------------------------
# 1. ADIM: lib/test/evaluation/local.py (Senin Windows yolu yazdığın dosya)
# Bunu Linux yollarıyla güncelliyoruz.
# ----------------------------------------------------------------
eval_local_path = os.path.join(base_path, 'lib', 'test', 'evaluation', 'local.py')
print(f"Düzeltiliyor: {eval_local_path}")

with open(eval_local_path, 'w') as f:
    f.write("from lib.test.evaluation.environment import EnvSettings\n\n")
    f.write("def local_env_settings():\n")
    f.write("    settings = EnvSettings()\n")
    f.write(f"    settings.davis_dir = ''\n")
    f.write(f"    settings.got10k_path = '{os.path.join(base_path, 'data', 'got10k')}'\n")
    f.write(f"    settings.lasot_path = '{os.path.join(base_path, 'data', 'lasot')}'\n")
    f.write(f"    settings.network_path = '{os.path.join(base_path, 'pretrained_models')}'\n") # Modellerin olduğu yer
    f.write(f"    settings.nfs_path = '{os.path.join(base_path, 'data', 'nfs')}'\n")
    f.write(f"    settings.otb_path = '{os.path.join(base_path, 'data', 'otb')}'\n")
    f.write(f"    settings.result_plot_path = '{os.path.join(base_path, 'output', 'test', 'result_plots')}'\n")
    f.write(f"    settings.results_path = '{os.path.join(base_path, 'output', 'test', 'tracking_results')}'\n")
    f.write(f"    settings.segmentation_path = '{os.path.join(base_path, 'output', 'test', 'segmentation_results')}'\n")
    f.write(f"    settings.trackingnet_path = '{os.path.join(base_path, 'data', 'trackingnet')}'\n")
    f.write(f"    settings.uav_path = '{os.path.join(base_path, 'data', 'uav')}'\n")
    f.write(f"    settings.vot_path = '{os.path.join(base_path, 'data', 'VOT2019')}'\n")
    f.write("    return settings\n")

# ----------------------------------------------------------------
# 2. ADIM: lib/train/admin/local.py (HATAYI VEREN DOSYA)
# PyTorch modeli yüklerken bu dosyayı arıyor. İçi boş olsa bile var olmalı.
# ----------------------------------------------------------------
train_admin_dir = os.path.join(base_path, 'lib', 'train', 'admin')
if not os.path.exists(train_admin_dir):
    os.makedirs(train_admin_dir)

train_local_path = os.path.join(train_admin_dir, 'local.py')
print(f"Oluşturuluyor (Hatayı çözecek kısım): {train_local_path}")

with open(train_local_path, 'w') as f:
    f.write("class EnvironmentSettings:\n")
    f.write("    def __init__(self):\n")
    f.write("        self.workspace_dir = ''\n")
    f.write("        self.tensorboard_dir = ''\n")
    f.write("        self.pretrained_networks = ''\n")
    f.write("        self.lasot_dir = ''\n")
    f.write("        self.got10k_dir = ''\n")
    f.write("        self.trackingnet_dir = ''\n")
    f.write("        self.coco_dir = ''\n")

print("\nBAŞARILI! Ayarlar Jetson/Linux ortamına göre güncellendi.")
print("Artık 'ortrack_tester.py' dosyasını çalıştırabilirsin.")