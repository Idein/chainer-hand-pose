from PIL import Image


def concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def paste_contain(canvas, src):
    canvasW, canvasH = canvas.size
    srcW, srcH = src.size
    scale_h = canvasW / srcW
    scale_w = canvasH / srcH
    scale = min(scale_h, scale_w)
    scaledW, scaledH = (int(srcW * scale), int(srcH * scale))
    resized = src.resize((scaledW, scaledH))
    offsetW = (canvasW - scaledW) // 2
    offsetH = (canvasH - scaledH) // 2
    canvas.paste(resized, (offsetW, offsetH))
    return canvas
