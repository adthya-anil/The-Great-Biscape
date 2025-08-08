import streamlit as st
import cv2
import numpy as np
import time

class BiscuitDunkerDetector:
    def __init__(self):
        self.roi_x1 = 100
        self.roi_y1 = 100
        self.roi_x2 = 500
        self.roi_y2 = 400
        self.green_strip_gap_cm = 1
        self.biscuit_length_cm = 5.6
        
        # Timer (for aggregate stats)
        self.is_dunking = False
        self.dunk_start_time = None
        self.total_dunk_time = 0

        # Fixed green strips
        self.fixed_green_strips = None
        self.is_strips_fixed = False
        self.target_strips_count = 7  

        # Calibration / smoothing
        self.pixels_per_cm = None
        self.last_depth_int = 0

    def detect_and_fix_green_strips(self, frame):
        if self.is_strips_fixed and self.fixed_green_strips is not None:
            return self.fixed_green_strips, None
        
        roi = frame[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        strips = []
        for c in contours:
            if cv2.contourArea(c) > 50:
                x, y, w, h = cv2.boundingRect(c)
                if w > h and w > 20:
                    strips.append((y, y + h))

        strips.sort(key=lambda x: x[0])
        
        if len(strips) >= self.target_strips_count and not self.is_strips_fixed:
            self.fixed_green_strips = strips[:self.target_strips_count]  
            self.is_strips_fixed = True
            return self.fixed_green_strips, mask
        
        return strips, mask

    def reset_green_strips_calibration(self):
        self.fixed_green_strips = None
        self.is_strips_fixed = False

    def detect_biscuit(self, frame):
        roi = frame[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_above = getattr(self, 'lower_biscuit', np.array([8, 60, 100]))
        upper_above = getattr(self, 'upper_biscuit', np.array([25, 255, 255]))

        lower_below = np.array([8, 30, 50])
        upper_below = np.array([25, 180, 200])

        mask_above = cv2.inRange(hsv, lower_above, upper_above)
        mask_below = cv2.inRange(hsv, lower_below, upper_below)

        mask = cv2.bitwise_or(mask_above, mask_below)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None, mask

        all_points = np.vstack(contours)
        hull = cv2.convexHull(all_points)
        return True, hull, mask

    def calculate_dunk_depth(self, biscuit_contour, green_strips):
        if biscuit_contour is None or len(green_strips) == 0:
            return 0

        if len(green_strips) >= 2:
            spacing = green_strips[1][0] - green_strips[0][0]
            if spacing > 0:
                self.pixels_per_cm = spacing / self.green_strip_gap_cm
        if not self.pixels_per_cm:
            self.pixels_per_cm = 15  # fallback

        biscuit_bottom = max(pt[0][1] for pt in biscuit_contour)
        water_surface = green_strips[0][0]
        if biscuit_bottom <= water_surface:
            return 0

        depth_cm = (biscuit_bottom - water_surface) / self.pixels_per_cm

        # Snap to integer with tolerance
        depth_int = int(round(depth_cm))
        if abs(depth_cm - self.last_depth_int) < 0.3:
            depth_int = self.last_depth_int
        else:
            self.last_depth_int = depth_int

        return max(0, depth_int)

    def update_timer(self, is_dunking_now):
        now = time.time()
        if is_dunking_now and not self.is_dunking:
            self.is_dunking = True
            self.dunk_start_time = now
        elif not is_dunking_now and self.is_dunking:
            self.is_dunking = False
            if self.dunk_start_time:
                self.total_dunk_time += now - self.dunk_start_time
            self.dunk_start_time = None

        return (now - self.dunk_start_time) if self.is_dunking and self.dunk_start_time else 0

    def draw_roi(self, frame):
        cv2.rectangle(frame, (self.roi_x1, self.roi_y1), (self.roi_x2, self.roi_y2), (255, 0, 0), 2)
        cv2.putText(frame, "Detection Area", (self.roi_x1, self.roi_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return frame

def main():
    st.set_page_config(page_title="Biscuit Dunk Timer", layout="wide")
    st.title("🍪 Biscuit Dunking Timer & Depth Detector")
    st.markdown("*Accurate, stable dunk depth & timing measurement with fixed green strips*")

    if 'detector' not in st.session_state:
        st.session_state.detector = BiscuitDunkerDetector()
    detector = st.session_state.detector

    # Biscuit safe/danger times (sec)
    safe_times = {1: 9, 2: 6, 3: 4, 4: 3}
    danger_times = {1: 12, 2: 8, 3: 7, 4: 5}

    # Debounce & reset tuning (adjust if you want more/less tolerance)
    DEPTH_DEBOUNCE_TIME = 0.35  # seconds a new depth must persist to be accepted
    ZERO_RESET_TIME = 0.6       # seconds of continuous zero-depth to consider biscuit removed

    # Session state for new behavior (initialize once)
    if 'global_dunk_start' not in st.session_state:
        st.session_state.global_dunk_start = None   # set when biscuit first touches water
    if 'debounced_depth' not in st.session_state:
        st.session_state.debounced_depth = 0        # depth after debounce
    if 'depth_candidate' not in st.session_state:
        st.session_state.depth_candidate = None
    if 'depth_change_since' not in st.session_state:
        st.session_state.depth_change_since = None
    if 'zero_since' not in st.session_state:
        st.session_state.zero_since = None
    if 'max_depth_reached' not in st.session_state:
        st.session_state.max_depth_reached = 0      # never decreases within a session

    with st.sidebar:
        st.header("🎛️ Controls")
        st.subheader("Detection Area")
        detector.roi_x1 = st.slider("Left", 0, 500, detector.roi_x1)
        detector.roi_y1 = st.slider("Top", 0, 400, detector.roi_y1)
        detector.roi_x2 = st.slider("Right", detector.roi_x1 + 50, 640, detector.roi_x2)
        detector.roi_y2 = st.slider("Bottom", detector.roi_y1 + 50, 480, detector.roi_y2)

        st.subheader("Green Strips Calibration")
        st.write(f"Target strips: {detector.target_strips_count}")
        strips_status = "✅ Fixed" if detector.is_strips_fixed else "❌ Not calibrated"
        st.write(f"Status: {strips_status}")
        
        if st.button("🔄 Reset Green Strips"):
            detector.reset_green_strips_calibration()
            st.success("Green strips calibration reset!")

        st.subheader("Biscuit Color Tuning")
        hue_min = st.slider("Hue Min", 0, 179, 8)
        sat_min = st.slider("Saturation Min", 0, 255, 60)
        val_min = st.slider("Brightness Min", 0, 255, 100)
        hue_max = st.slider("Hue Max", hue_min, 179, 25)
        sat_max = st.slider("Saturation Max", sat_min, 255, 255)
        val_max = st.slider("Brightness Max", val_min, 255, 255)

        if st.button("🔄 Reset Timer / Session"):
            detector.is_dunking = False
            detector.dunk_start_time = None
            detector.total_dunk_time = 0
            # reset our new session vars
            st.session_state.global_dunk_start = None
            st.session_state.debounced_depth = 0
            st.session_state.depth_candidate = None
            st.session_state.depth_change_since = None
            st.session_state.zero_since = None
            st.session_state.max_depth_reached = 0
            st.success("Timer & session reset!")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("📹 Live Camera Feed")
        camera_placeholder = st.empty()
    with col2:
        st.header("📊 Measurements")
        depth_placeholder = st.empty()
        timer_placeholder = st.empty()
        status_placeholder = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera")
                break
            frame = cv2.flip(frame, 1)

            detector.lower_biscuit = np.array([hue_min, sat_min, val_min])
            detector.upper_biscuit = np.array([hue_max, sat_max, val_max])

            frame_overlay = detector.draw_roi(frame.copy())
            biscuit_detected, biscuit_contour, _ = detector.detect_biscuit(frame)
            green_strips, _ = detector.detect_and_fix_green_strips(frame)

            dunk_depth = 0
            if biscuit_detected and green_strips:
                dunk_depth = detector.calculate_dunk_depth(biscuit_contour, green_strips)

            # Keep detector's own dunk timer (for total session time)
            is_currently_dunking = dunk_depth > 0
            current_session = detector.update_timer(is_currently_dunking)

            # ----------------------
            # New robust countdown logic
            # ----------------------
            now = time.time()

            if dunk_depth > 0:
                # clear zero/remove timer
                st.session_state.zero_since = None

                # start global dunk start time (only once per dunk session)
                if st.session_state.global_dunk_start is None:
                    st.session_state.global_dunk_start = now

                # debounce candidate depth changes
                if st.session_state.depth_candidate is None or dunk_depth != st.session_state.depth_candidate:
                    st.session_state.depth_candidate = dunk_depth
                    st.session_state.depth_change_since = now
                else:
                    # candidate persists — upgrade to debounced depth after debounce time
                    if st.session_state.debounced_depth != st.session_state.depth_candidate:
                        if now - (st.session_state.depth_change_since or now) >= DEPTH_DEBOUNCE_TIME:
                            st.session_state.debounced_depth = st.session_state.depth_candidate
                            # update max depth reached — only increases during a session
                            if st.session_state.debounced_depth > st.session_state.max_depth_reached:
                                st.session_state.max_depth_reached = st.session_state.debounced_depth

                # active depth for countdown = max depth reached so far (never lowers mid-session)
                active_depth = st.session_state.max_depth_reached if st.session_state.max_depth_reached > 0 else st.session_state.debounced_depth

                if st.session_state.global_dunk_start is not None and active_depth in safe_times:
                    elapsed = now - st.session_state.global_dunk_start
                    safe_remaining = max(0.0, safe_times[active_depth] - elapsed)
                    danger_remaining = max(0.0, danger_times[active_depth] - elapsed)
                else:
                    safe_remaining = 0.0
                    danger_remaining = 0.0

            else:
                # detected depth == 0 (not in water)
                if st.session_state.global_dunk_start is None:
                    # nothing to do
                    st.session_state.depth_candidate = None
                    st.session_state.debounced_depth = 0
                    st.session_state.max_depth_reached = 0
                    safe_remaining = 0.0
                    danger_remaining = 0.0
                else:
                    # start zero persistence timer; only reset session if zero is stable for ZERO_RESET_TIME
                    if st.session_state.zero_since is None:
                        st.session_state.zero_since = now
                        safe_remaining = 0.0
                        danger_remaining = 0.0
                    else:
                        if now - st.session_state.zero_since >= ZERO_RESET_TIME:
                            # consider biscuit removed — reset session
                            st.session_state.global_dunk_start = None
                            st.session_state.debounced_depth = 0
                            st.session_state.depth_candidate = None
                            st.session_state.depth_change_since = None
                            st.session_state.zero_since = None
                            st.session_state.max_depth_reached = 0
                            safe_remaining = 0.0
                            danger_remaining = 0.0
                        else:
                            # still within grace period — show last known remaining based on max_depth_reached
                            active_depth = st.session_state.max_depth_reached
                            if active_depth in safe_times and st.session_state.global_dunk_start:
                                elapsed = now - st.session_state.global_dunk_start
                                safe_remaining = max(0.0, safe_times[active_depth] - elapsed)
                                danger_remaining = max(0.0, danger_times[active_depth] - elapsed)
                            else:
                                safe_remaining = 0.0
                                danger_remaining = 0.0

            # ----------------------
            # Draw and display UI
            # ----------------------
            if biscuit_detected and biscuit_contour is not None:
                cv2.drawContours(frame_overlay, [biscuit_contour], -1, (0, 0, 255), 2,
                                 offset=(detector.roi_x1, detector.roi_y1))

            for i, (top, _) in enumerate(green_strips):
                y1 = detector.roi_y1 + top
                color = (0, 255, 0) if detector.is_strips_fixed else (0, 255, 255)
                line_thickness = 3 if detector.is_strips_fixed else 2
                cv2.line(frame_overlay, (detector.roi_x1, y1), (detector.roi_x2, y1), color, line_thickness)
                label = "Water Level (0cm)" if i == 0 else f"{i}cm deep"
                cv2.putText(frame_overlay, label, (detector.roi_x2 + 5, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            if dunk_depth > 0:
                cv2.putText(frame_overlay, f"Depth: {dunk_depth} cm", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            if detector.is_strips_fixed:
                cv2.putText(frame_overlay, "GREEN STRIPS FIXED", (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame_overlay, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            with depth_placeholder.container():
                st.metric("🎯 Dunk Depth (raw)", f"{dunk_depth} cm")
                st.metric("🧾 Debounced Depth", f"{st.session_state.debounced_depth} cm")
                st.metric("🔼 Max Depth Reached", f"{st.session_state.max_depth_reached} cm")

            with timer_placeholder.container():
                st.metric("✅ Safe Time Left", f"{safe_remaining:.1f}s")
                st.metric("⚠️ Danger Time Left", f"{danger_remaining:.1f}s")
                total_time = detector.total_dunk_time + (current_session if current_session else 0)
                st.metric("🕒 Total Dunk Time", f"{total_time:.1f}s")

            # Status messaging
            if not biscuit_detected:
                status = "❌ Position biscuit in detection area"
            elif not detector.is_strips_fixed:
                status = f"🔍 Calibrating green strips... Found {len(green_strips)}/{detector.target_strips_count}"
            elif not green_strips:
                status = "❌ Green strips calibration lost - reset needed"
            elif st.session_state.global_dunk_start is None or st.session_state.max_depth_reached == 0:
                status = "✅ Ready to measure - lower biscuit into water"
            else:
                if safe_remaining <= 0 < danger_remaining:
                    status = "⚠️ WARNING: Biscuit near breaking point!"
                elif danger_remaining <= 0:
                    status = "💥 BISCUIT FAILURE IMMINENT!"
                else:
                    status = f"🎯 Dunking! active depth {st.session_state.max_depth_reached}cm"
            status_placeholder.markdown(f"**Status:** {status}")

            time.sleep(0.03)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

if __name__ == "__main__":
    main()
