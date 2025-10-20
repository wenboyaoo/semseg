import os

root= "dataset/voc2012"
split_dir = os.path.join(root,"ImageSets/Segmentation")
output_dir = os.path.join(root,"list")

os.makedirs(output_dir,exist_ok=True)
for split in ["train","val"]:
    txt_in = os.path.join(split_dir,f"{split}.txt")
    txt_out = os.path.join(output_dir,f"{split}.txt")

    with open(txt_in, "r") as f:
        names = [x.strip() for x in f.readlines() if x.strip()]
    
    with open(txt_out, "w") as f:
        for name in names:
            img_path = os.path.join("JPEGImages", f"{name}.jpg")
            label_path = os.path.join("SegmentationClass", f"{name}.png")
            f.write(f"{img_path} {label_path}\n")

    print(f"{split} list saved to {txt_out}")