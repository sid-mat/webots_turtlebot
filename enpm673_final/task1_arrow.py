import cv2
import numpy as np
from collections import deque

class ArrowDetector:
    def __init__(self):
        self.last_offset = 0.0
        self.offset_buffer = deque(maxlen=5)
        self.no_arrow_count = 0
        self.NO_ARROW_LIMIT = 45

    def process(self, frame):
        h, w = frame.shape[:2]
        display = frame.copy()

        roi_top = int(h * 0.55)
        roi = frame[roi_top:, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        best_bottom = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3000 or area > 200000:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw > w * 0.85:
                continue
            if bh < 20:
                continue
            aspect = bw / float(bh)
            if aspect > 5.0:
                continue
            bottom = y + bh
            if bottom > best_bottom:
                best_bottom = bottom
                best_cnt = cnt

        if best_cnt is None:
            self.no_arrow_count += 1
            self.offset_buffer.clear()
            done = self.no_arrow_count >= self.NO_ARROW_LIMIT

            if done:
                return None, None, None, display, True

            # Drive straight when no paper found — don't use last offset
            return 0.0, 0.0, None, display, False

        x, y, bw, bh = cv2.boundingRect(best_cnt)
        y_full = y + roi_top

        # Show the captured region
        crop_debug = roi[max(0,y):min(roi.shape[0],y+bh),
                        max(0,x):min(roi.shape[1],x+bw)]
        if crop_debug.size > 0:
            cv2.imshow("Captured Paper", crop_debug)

        arrow_valid = self._validate_arrow(roi, x, y, bw, bh)

        if not arrow_valid:
            self.no_arrow_count += 1
            done = self.no_arrow_count >= self.NO_ARROW_LIMIT

            cv2.rectangle(display,
                        (x, y_full), (x + bw, y_full + bh),
                        (0, 0, 255), 2)
            cv2.putText(display, "NO ARROW",
                        (x, max(y_full - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if done:
                return None, None, None, display, True

            # Drive straight — don't use last offset
            return 0.0, 0.0, None, display, False

        self.no_arrow_count = 0

        arrow_center_x = x + bw // 2
        frame_center_x = w // 2
        raw_offset = (arrow_center_x - frame_center_x) / float(frame_center_x)

        self.offset_buffer.append(raw_offset)
        smoothed_offset = float(np.mean(self.offset_buffer))
        self.last_offset = smoothed_offset

        cv2.rectangle(display,
                      (x, y_full), (x + bw, y_full + bh),
                      (0, 255, 0), 3)
        cv2.putText(display,
                    f"OFFSET: {smoothed_offset:.2f}",
                    (x, max(y_full - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.line(display, (w // 2, roi_top),
                 (w // 2, h), (255, 0, 0), 1)
        cv2.line(display, (arrow_center_x, y_full),
                 (arrow_center_x, y_full + bh), (0, 255, 255), 2)

        bbox = (x, y_full, bw, bh)
        return smoothed_offset, smoothed_offset, bbox, display, False

    def _validate_arrow(self, roi, x, y, bw, bh):
        # crop to only white region
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(roi.shape[1], x + bw)
        y2 = min(roi.shape[0], y + bh)

        crop = roi[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        ch, cw = crop.shape[:2]
        crop_area = ch * cw

        # not paper conditioin
        if crop_area > 150000:
            return False

        crop_aspect = cw / float(ch) if ch > 0 else 0
        if crop_aspect > 6.0:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        white_ratio = np.sum(normalized > 140) / float(crop_area)
        if white_ratio < 0.10:  # was 0.15
            return False

        # find black region
        _, black = cv2.threshold(normalized, 80, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            black, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False

        if len(contours) > 20:
            return False

        arrow_sized = 0
        total_black_area = 0
        chunky_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            total_black_area += area
            if crop_area * 0.005 < area < crop_area * 0.60:  # looser both ends
                arrow_sized += 1
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                cnt_aspect = rw / float(rh) if rh > 0 else 0
                if 0.10 < cnt_aspect < 10.0:  # looser aspect
                    chunky_count += 1

        black_ratio = total_black_area / float(crop_area)

        # Only need 1 valid contour
        if arrow_sized < 1:
            return False

        # Minimum black — very relaxed for far-away papers
        if black_ratio < 0.02:  # was 0.05
            return False

        # Max black — keep to reject pure floor
        if black_ratio > 0.85:
            return False

        # Must have at least some white — confirms it's paper
        if white_ratio < 0.30:
            return False

        return True