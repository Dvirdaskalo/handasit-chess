import os
import shutil
import random

def check_and_fix_data_issues():
    """Check for common data issues and fix them"""
    
    train_dir = "data/train"
    
    print("Checking data issues...")
    
    # 1. Check if we have wr (white rook) directory
    wr_dir = os.path.join(train_dir, "wr")
    wp_dir = os.path.join(train_dir, "wp")
    
    if not os.path.exists(wr_dir):
        print("ERROR: No 'wr' directory found!")
        return False
    
    # 2. Count images
    wr_count = len([f for f in os.listdir(wr_dir) if f.endswith('.png')])
    wp_count = len([f for f in os.listdir(wp_dir) if f.endswith('.png')])
    
    print(f"Current counts: wp={wp_count}, wr={wr_count}")
    
    if wr_count < wp_count * 0.5:
        print(f"WARNING: Not enough rook images! (Have {wr_count}, need at least {int(wp_count * 0.8)})")
        
        # Option 1: Copy some pawn images and rename them (as a temporary fix)
        # This is NOT recommended for production, but for testing
        response = input("Do you want to create synthetic rook data from pawns? (y/n): ")
        if response.lower() == 'y':
            # Get some pawn images
            pawn_images = [f for f in os.listdir(wp_dir) if f.endswith('.png')]
            num_to_copy = min(50, len(pawn_images) // 2)
            
            print(f"Creating {num_to_copy} synthetic rook images...")
            for i in range(num_to_copy):
                src = os.path.join(wp_dir, random.choice(pawn_images))
                dst = os.path.join(wr_dir, f"wr_synth_{i:04d}.png")
                shutil.copy(src, dst)
            
            print(f"Created {num_to_copy} synthetic images in wr directory")
    
    # 3. Check image sizes
    print("\nChecking image sizes...")
    for cls in ["wr", "wp"]:
        cls_dir = os.path.join(train_dir, cls)
        images = [f for f in os.listdir(cls_dir) if f.endswith('.png')][:5]  # Check first 5
        
        for img_file in images:
            img_path = os.path.join(cls_dir, img_file)
            try:
                import cv2
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  WARNING: Cannot read {cls}/{img_file}")
                else:
                    print(f"  {cls}/{img_file}: {img.shape}")
            except:
                print(f"  ERROR: Failed to check {cls}/{img_file}")
    
    return True

def create_test_setup():
    """Create a simple test setup to verify piece recognition"""
    
    print("\nCreating test setup...")
    
    # Create a directory with clear test images
    test_dir = "test_setup"
    os.makedirs(test_dir, exist_ok=True)
    
    # Instructions
    print(f"""
To test your model properly:
1. Place a white rook on a dark square (like a1)
2. Place a white pawn on a light square (like b2)
3. Place nothing on a3 (empty square)
4. Take photos from the same angle as training

Save the images as:
  {test_dir}/rook_on_dark.png
  {test_dir}/pawn_on_light.png
  {test_dir}/empty_square.png

Then run the diagnostic script to see if the model can distinguish them.
""")
    
    return test_dir

if __name__ == "__main__":
    check_and_fix_data_issues()
    create_test_setup()
    print("\nNext steps:")
    print("1. Collect more clear images of white rooks from different angles")
    print("2. Make sure rook images show the distinct rook shape (crenelations)")
    print("3. Try reducing the number of classes (train only on: empty, wp, wr)")
    print("4. Increase training epochs to 50+")
    print("5. Lower confidence threshold to 0.5 in infer.py")