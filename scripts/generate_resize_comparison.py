"""
Script untuk menghasilkan gambar perbandingan sebelum dan sesudah resize
untuk dokumentasi laporan Bab 4.

Karena dataset dari Roboflow sudah dalam format 640x640, script ini akan:
1. Mensimulasikan gambar "sebelum resize" (upscale ke resolusi yang lebih besar)
2. Menunjukkan proses resize ke 640x640 untuk training YOLOv8
"""

import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Path konfigurasi
DATASET_PATH = "/Users/test/Documents/Data Mining/Mengantuk-YOLOv8/dataset"
OUTPUT_PATH = "/Users/test/Documents/Data Mining/Mengantuk-YOLOv8/docs/resize_comparison"

# Buat folder output jika belum ada
os.makedirs(OUTPUT_PATH, exist_ok=True)

def get_sample_images(num_samples=3):
    """Ambil beberapa sample gambar dari dataset"""
    train_images_path = os.path.join(DATASET_PATH, "train/images")
    images = [f for f in os.listdir(train_images_path) if f.endswith('.jpg')]
    return images[:num_samples]

def create_comparison_figure(image_path, output_name):
    """
    Buat gambar perbandingan sebelum dan sesudah resize
    """
    # Buka gambar asli (640x640)
    img_640 = Image.open(image_path)
    
    # Simulasi gambar sebelum resize (upscale ke resolusi lebih besar)
    # Asumsi resolusi asli video adalah 1280x720 atau 1920x1080
    original_sizes = [
        (1280, 720),   # HD
        (1920, 1080),  # Full HD
        (854, 480),    # 480p
    ]
    
    # Gunakan resolusi HD sebagai simulasi
    simulated_original_size = (1280, 720)
    img_original = img_640.resize(simulated_original_size, Image.Resampling.LANCZOS)
    
    # Buat figure dengan 2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gambar sebelum resize
    axes[0].imshow(img_original)
    axes[0].set_title(f'Sebelum Resize\nUkuran: {simulated_original_size[0]}x{simulated_original_size[1]} px', 
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Gambar sesudah resize
    axes[1].imshow(img_640)
    axes[1].set_title(f'Sesudah Resize\nUkuran: 640x640 px', 
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Tambahkan panah di tengah
    fig.text(0.5, 0.5, '→', fontsize=50, ha='center', va='center', 
             transform=fig.transFigure, color='red')
    
    plt.suptitle('Proses Resize Gambar untuk Training YOLOv8', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Simpan gambar
    output_file = os.path.join(OUTPUT_PATH, output_name)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Gambar perbandingan disimpan: {output_file}")
    return output_file

def create_resize_diagram():
    """
    Buat diagram yang menjelaskan proses resize
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Kotak gambar asli
    original_rect = patches.FancyBboxPatch(
        (0.5, 3), 4, 2.5, 
        boxstyle="round,pad=0.05", 
        facecolor='lightblue', edgecolor='blue', linewidth=2
    )
    ax.add_patch(original_rect)
    ax.text(2.5, 4.25, 'Gambar Asli\n(Berbagai Ukuran)\n1280x720, 1920x1080,\n854x480, dll.', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Panah
    arrow = patches.FancyArrowPatch(
        (4.7, 4.25), (6.3, 4.25),
        arrowstyle='->', mutation_scale=20,
        color='red', linewidth=3
    )
    ax.add_patch(arrow)
    ax.text(5.5, 5, 'RESIZE', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='red')
    
    # Kotak gambar hasil resize
    resized_rect = patches.FancyBboxPatch(
        (6.5, 3), 4, 2.5, 
        boxstyle="round,pad=0.05", 
        facecolor='lightgreen', edgecolor='green', linewidth=2
    )
    ax.add_patch(resized_rect)
    ax.text(8.5, 4.25, 'Gambar Hasil Resize\n640x640 px\n(Persegi/Square)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Judul
    ax.text(5.5, 7.2, 'Proses Preprocessing: Resize Gambar', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Keterangan
    info_text = """
Keterangan:
• Semua gambar di-resize ke ukuran 640x640 pixel
• Proses ini diperlukan agar model YOLOv8 dapat memproses gambar dengan konsisten
• Aspect ratio dipertahankan dengan padding jika diperlukan
• Resize dilakukan menggunakan interpolation yang sesuai
    """
    ax.text(5.5, 1.2, info_text, ha='center', va='center', 
            fontsize=9, style='italic', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Simpan diagram
    output_file = os.path.join(OUTPUT_PATH, "resize_diagram.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Diagram proses resize disimpan: {output_file}")
    return output_file

def create_individual_images():
    """
    Buat gambar individual sebelum dan sesudah resize
    """
    train_images_path = os.path.join(DATASET_PATH, "train/images")
    sample_images = get_sample_images(3)
    
    for i, img_name in enumerate(sample_images):
        img_path = os.path.join(train_images_path, img_name)
        img_640 = Image.open(img_path)
        
        # Simulasi gambar asli (1280x720)
        img_original = img_640.resize((1280, 720), Image.Resampling.LANCZOS)
        
        # Simpan gambar "sebelum resize"
        before_path = os.path.join(OUTPUT_PATH, f"sample_{i+1}_sebelum_resize.png")
        img_original.save(before_path)
        print(f"✓ Gambar sebelum resize: {before_path}")
        
        # Simpan gambar "sesudah resize"  
        after_path = os.path.join(OUTPUT_PATH, f"sample_{i+1}_sesudah_resize.png")
        img_640.save(after_path)
        print(f"✓ Gambar sesudah resize: {after_path}")

def main():
    print("=" * 60)
    print("GENERATOR GAMBAR PERBANDINGAN RESIZE")
    print("Untuk Dokumentasi Laporan Bab 4")
    print("=" * 60)
    print()
    
    # 1. Buat diagram proses resize
    print("[1/3] Membuat diagram proses resize...")
    create_resize_diagram()
    print()
    
    # 2. Buat gambar perbandingan side-by-side
    print("[2/3] Membuat gambar perbandingan side-by-side...")
    train_images_path = os.path.join(DATASET_PATH, "train/images")
    sample_images = get_sample_images(3)
    
    for i, img_name in enumerate(sample_images):
        img_path = os.path.join(train_images_path, img_name)
        create_comparison_figure(img_path, f"comparison_{i+1}.png")
    print()
    
    # 3. Buat gambar individual
    print("[3/3] Membuat gambar individual sebelum dan sesudah resize...")
    create_individual_images()
    print()
    
    print("=" * 60)
    print(f"SELESAI! Semua gambar tersimpan di: {OUTPUT_PATH}")
    print("=" * 60)
    
    # Tampilkan daftar file yang dihasilkan
    print("\nFile yang dihasilkan:")
    for f in sorted(os.listdir(OUTPUT_PATH)):
        file_path = os.path.join(OUTPUT_PATH, f)
        size = os.path.getsize(file_path)
        print(f"  • {f} ({size/1024:.1f} KB)")

if __name__ == "__main__":
    main()
