import cv2
import numpy as np
import mediapipe as mp
import time, random, math, sys

# --------------------------
# Config
# --------------------------
BG_CAPTURE_FRAMES = 35
BG_COUNTDOWN_SECS = 3
ERASER_RADIUS = 80
FEATHER_KERNEL = 61
SMOOTHING = 0.75
MAX_LIGHTNING_PER_FRAME = 3
PARTICLE_SPAWN = 6
PARTICLE_LIFE = 20

mp_hands = mp.solutions.hands

# --------------------------
# Camera helper
# --------------------------
def try_open_camera(max_try=3):
    for i in range(max_try):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[INFO] Camera {i} opened {frame.shape[1]}x{frame.shape[0]}")
                return cap
        cap.release()
    print("[ERROR] No camera found.")
    return None

# --------------------------
# Background capture
# --------------------------
def capture_background(cap, frames=BG_CAPTURE_FRAMES, countdown=BG_COUNTDOWN_SECS):
    print("[INFO] Capturing background... Step out of the frame.")
    time.sleep(1)
    buf = []
    for i in range(frames):
        ret, f = cap.read()
        if not ret:
            continue
        f = cv2.flip(f, 1)
        buf.append(f.astype(np.float32))
        show = (np.median(buf, axis=0)).astype(np.uint8)
        cv2.putText(show, f"Capturing background... {i+1}/{frames}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Capture Preview", show)
        cv2.waitKey(1)
    cv2.destroyWindow("Capture Preview")
    return np.median(np.stack(buf, axis=0), axis=0).astype(np.uint8)

# --------------------------
# Particles + lightning
# --------------------------
def spawn_particles(particles, x, y, count=8):
    for _ in range(count):
        vx, vy = random.uniform(-1.5, 1.5), random.uniform(-3.0, -0.5)
        size = random.randint(2, 6)
        life = random.randint(int(PARTICLE_LIFE*0.6), PARTICLE_LIFE)
        color = random.choice([(255,255,0),(200,200,255),(255,120,255)])
        particles.append({'pos':[x,y],'vel':[vx,vy],'life':life,'size':size,'color':color})

def update_and_draw_particles(frame, particles):
    new_list = []
    for p in particles:
        if p['life'] <= 0: continue
        p['pos'][0] += p['vel'][0]
        p['pos'][1] += p['vel'][1]
        p['vel'][1] += 0.12
        x,y = int(p['pos'][0]), int(p['pos'][1])
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (x,y), p['size'], p['color'], -1)
        p['life'] -= 1
        if p['life'] > 0: new_list.append(p)
    particles[:] = new_list

def draw_lightning_bolts(frame, x, y, count=2):
    for _ in range(count):
        length = random.randint(40, 100)
        angle = random.uniform(-math.pi, math.pi)
        ex, ey = int(x + length*math.cos(angle)), int(y + length*math.sin(angle))
        pts = [(x,y)]
        for i in range(1, random.randint(3,5)):
            t = i/(random.randint(3,5))
            pts.append((int(x+(ex-x)*t+random.randint(-15,15)),
                        int(y+(ey-y)*t+random.randint(-15,15))))
        pts_np = np.array(pts, np.int32)
        cv2.polylines(frame,[pts_np],False,(255,255,255),thickness=2)

# --------------------------
# Main
# --------------------------
def main():
    cap = try_open_camera()
    if cap is None: sys.exit(1)

    for _ in range(10): cap.read()
    background = capture_background(cap)

    h, w = background.shape[:2]
    eraser_mask = np.zeros((h,w),dtype=np.uint8)
    hands = mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.6,min_tracking_confidence=0.6)

    smoothed, particles = None,[]
    erasing_enabled = False

    # Button region
    btn_w, btn_h = 200, 60
    btn_x1, btn_y1 = w - btn_w - 20, 20
    btn_x2, btn_y2 = btn_x1 + btn_w, btn_y1 + btn_h
    pulse_phase = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        frame = cv2.resize(frame,(w,h))

        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        fingertip = None
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            tip_x, tip_y = int(lm[8].x*w), int(lm[8].y*h)
            fingertip = (tip_x, tip_y)

            # Button check
            if not erasing_enabled:
                if btn_x1 < tip_x < btn_x2 and btn_y1 < tip_y < btn_y2:
                    erasing_enabled = True
                    print("[INFO] Magic Eraser Activated!")

        # Draw button


        
        if not erasing_enabled:
            pulse_phase += 0.2
            intensity = (math.sin(pulse_phase)+1)/2
            color = (0, int(150+100*intensity), int(255*intensity))
            cv2.rectangle(frame,(btn_x1,btn_y1),(btn_x2,btn_y2),color,-1)
            cv2.putText(frame,"START MAGIC",(btn_x1+20,btn_y1+40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2)

        if erasing_enabled and fingertip:
            if smoothed is None:
                smoothed = np.array(fingertip,dtype=np.float32)
            else:
                smoothed = smoothed*SMOOTHING+np.array(fingertip,dtype=np.float32)*(1-SMOOTHING)
            sx,sy = int(smoothed[0]),int(smoothed[1])
            cv2.circle(eraser_mask,(sx,sy),ERASER_RADIUS,255,-1)
            spawn_particles(particles,sx,sy,PARTICLE_SPAWN)
            if random.random()<0.4: draw_lightning_bolts(frame,sx,sy)

        # Apply erase
        k = FEATHER_KERNEL if FEATHER_KERNEL%2==1 else FEATHER_KERNEL+1
        mask_blur = cv2.GaussianBlur(eraser_mask,(k,k),0)
        alpha = mask_blur.astype(np.float32)/255.0
        alpha_3 = cv2.merge([alpha,alpha,alpha])
        out = (frame.astype(np.float32)*(1-alpha_3)+background.astype(np.float32)*alpha_3).astype(np.uint8)

        update_and_draw_particles(out,particles)
        cv2.imshow("Magic Eraser",out)

        key=cv2.waitKey(1)&0xFF
        if key in [ord('q'),27]: break
        elif key==ord('c'): eraser_mask[:]=0;particles.clear();erasing_enabled=False
        elif key==ord('r'): background=capture_background(cap);eraser_mask[:]=0;erasing_enabled=False

    cap.release();hands.close();cv2.destroyAllWindows()

if __name__=="__main__":
    main()
