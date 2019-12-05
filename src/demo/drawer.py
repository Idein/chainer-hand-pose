from PIL import ImageDraw


def draw_axis(pil_img, y_offset, x_offset, axisH, axisW, h_color, w_color, h_text, w_text):
    drawer = ImageDraw.Draw(pil_img)
    lw = 5
    ax_offset = lw // 2
    length_offset = 10
    # draw horizontal line
    drawer.line(
        (
            x_offset + lw,
            y_offset + ax_offset,
            x_offset + axisW - length_offset,
            y_offset + ax_offset,
        ),
        fill=w_color,
        width=lw
    )
    # draw vertical line
    drawer.line(
        (
            x_offset + ax_offset,
            y_offset + lw,
            x_offset + ax_offset,
            y_offset + axisH - length_offset,
        ),
        fill=h_color,
        width=lw
    )
    text_horizontal_x_offset = 7
    text_horizontal_y_offset = 4
    text_y_offset = 15
    # write horizontal axis
    drawer.text(
        (x_offset + axisW - text_horizontal_x_offset, y_offset - text_horizontal_y_offset),
        w_text,
        fill=(255, 255, 255)
    )
    # write vertical axis
    drawer.text(
        (x_offset, y_offset + axisH - text_y_offset),
        h_text,
        fill=(255, 255, 255)
    )


def draw_hands(pil_img, kp_vu, edges, color_map):
    drawer = ImageDraw.Draw(pil_img)
    r = 2
    for p in kp_vu:
        # yx yx ... -> xy xy ...
        for i, (x, y) in enumerate(p[:, ::-1]):
            color = tuple(color_map[i])
            drawer.ellipse((x - r, y - r, x + r, y + r),
                           fill=color)
        for s, t in edges:
            sy, sx = p[s].astype(int)
            ty, tx = p[t].astype(int)
            color = tuple(color_map[s, t])
            drawer.line([(sx, sy), (tx, ty)], fill=color)


def draw_bbox(pil_img, bbox, label, hand_class, color_map=None):
    drawer = ImageDraw.Draw(pil_img)
    for b, l in zip(bbox, label):
        ymin, xmin, ymax, xmax = b.astype(int)
        name = hand_class[l]
        if color_map is None:
            color = (128, 128, 128)
        else:
            color = color_map[name]
        drawer.rectangle(
            xy=[xmin, ymin, xmax, ymax],
            fill=None,
            outline=color
        )
