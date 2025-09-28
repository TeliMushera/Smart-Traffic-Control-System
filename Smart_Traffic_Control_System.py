import torch
import cv2
import numpy as np

# ==== PARAMETERS ====
BASE_GREEN = 5
SCALE_PER_VEHICLE = 2
MAX_GREEN = 20
YELLOW_TIME = 3
ALL_RED = 1

def compute_green_time(count):
    return min(BASE_GREEN + count * SCALE_PER_VEHICLE, MAX_GREEN)

def get_rois(frame):
    h, w = frame.shape[:2]
    # NS lane (vertical)
    roi_ns = (int(w*0.40), 0, int(w*0.2), int(h*0.7))
    # EW lane (horizontal)
    roi_ew = (0, int(h*0.40), int(w*0.7), int(h*0.2))
    return roi_ns, roi_ew

def count_vehicles(dets, roi):
    x, y, w, h = roi
    c = 0
    for d in dets:
        cls = int(d[5])
        if cls in [2,3,5,7]:  # vehicle classes
            x1,y1,x2,y2 = d[:4]
            cx, cy = (x1+x2)/2, (y1+y2)/2
            if x <= cx <= x+w and y <= cy <= y+h:
                c += 1
    return c

def draw_signal(frame, state, countdown, pos, label):
    """Draw traffic light at given pos=(x,y)"""
    x,y = pos
    cv2.rectangle(frame, (x, y), (x+60, y+150), (50,50,50), -1)
    colors = {"red":(0,0,255), "yellow":(0,255,255), "green":(0,255,0)}
    # default all off
    c_red, c_yel, c_grn = (80,80,80),(80,80,80),(80,80,80)
    if state=="GREEN": c_grn = colors["green"]
    elif state=="YELLOW": c_yel = colors["yellow"]
    else: c_red = colors["red"]
    cv2.circle(frame,(x+30,y+30),20,c_red,-1)
    cv2.circle(frame,(x+30,y+75),20,c_yel,-1)
    cv2.circle(frame,(x+30,y+120),20,c_grn,-1)
    cv2.putText(frame,f"{label}: {int(countdown)}s",(x-10,y+175),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

# ==== LOAD YOLO ====
model = torch.hub.load('./','custom',path='yolov5s.pt',source='local')

# ==== INPUT VIDEO PATH ====
video_path = "input.mp4"   # <<< put your video file here
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
w,h = int(cap.get(3)), int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('smart_output.mp4', fourcc, fps, (w,h))

# ==== CONTROLLER STATE MACHINE ====
state = "NS_GREEN"
state_start = 0
green_ns, green_ew = BASE_GREEN, BASE_GREEN
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000  # time in sec

    roi_ns, roi_ew = get_rois(frame)
    results = model(frame, size=640)
    dets = results.xyxy[0].cpu().numpy()
    frame = np.squeeze(results.render())

    cnt_ns = count_vehicles(dets, roi_ns)
    cnt_ew = count_vehicles(dets, roi_ew)

    green_ns = compute_green_time(cnt_ns)
    green_ew = compute_green_time(cnt_ew)

    elapsed = t - state_start

    # ==== STATE TRANSITIONS ====
    if state=="NS_GREEN" and elapsed>=green_ns:
        state="NS_YELLOW"; state_start=t
    elif state=="NS_YELLOW" and elapsed>=YELLOW_TIME:
        state="ALL_RED"; state_start=t; last="NS"
    elif state=="EW_GREEN" and elapsed>=green_ew:
        state="EW_YELLOW"; state_start=t
    elif state=="EW_YELLOW" and elapsed>=YELLOW_TIME:
        state="ALL_RED"; state_start=t; last="EW"
    elif state=="ALL_RED" and elapsed>=ALL_RED:
        state="EW_GREEN" if last=="NS" else "NS_GREEN"
        state_start=t

    # ==== DRAW ROIs ====
    cv2.rectangle(frame,(roi_ns[0],roi_ns[1]),
                  (roi_ns[0]+roi_ns[2],roi_ns[1]+roi_ns[3]),(0,255,0),2)
    cv2.rectangle(frame,(roi_ew[0],roi_ew[1]),
                  (roi_ew[0]+roi_ew[2],roi_ew[1]+roi_ew[3]),(255,0,0),2)

    # ==== DRAW SIGNALS ====
    if state.startswith("NS"):
        countdown = green_ns-elapsed if state=="NS_GREEN" else YELLOW_TIME-elapsed
        draw_signal(frame,"GREEN" if state=="NS_GREEN" else "YELLOW" if state=="NS_YELLOW" else "RED",
                    countdown,(50,50),"NS")
        draw_signal(frame,"RED",0,(w-100,50),"EW")
    elif state.startswith("EW"):
        countdown = green_ew-elapsed if state=="EW_GREEN" else YELLOW_TIME-elapsed
        draw_signal(frame,"GREEN" if state=="EW_GREEN" else "YELLOW" if state=="EW_YELLOW" else "RED",
                    countdown,(w-100,50),"EW")
        draw_signal(frame,"RED",0,(50,50),"NS")
    else: # ALL_RED
        draw_signal(frame,"RED",ALL_RED-elapsed,(50,50),"NS")
        draw_signal(frame,"RED",ALL_RED-elapsed,(w-100,50),"EW")

    # ==== VEHICLE COUNTS ====
    cv2.putText(frame,f"NS Vehicles: {cnt_ns} | Green: {green_ns}s",(10,h-40),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.putText(frame,f"EW Vehicles: {cnt_ew} | Green: {green_ew}s",(10,h-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

    out.write(frame)
    frame_idx+=1

cap.release()
out.release()
print("✅ Done! Video saved as smart_output.mp4")
