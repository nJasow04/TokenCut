import PIL.Image as Image 

def resize_pil(I, patch_size=16, resize=None) : 
    if resize is None:
        w, h = I.size
    else:
        h = resize[0]
        w = resize[1]
        I = I.resize((w, h))

    new_w, new_h = int(round(w / patch_size)) * patch_size, int(round(h / patch_size)) * patch_size
    feat_w, feat_h = new_w // patch_size, new_h // patch_size

    return I.resize((new_w, new_h), resample=Image.LANCZOS), w, h, feat_w, feat_h