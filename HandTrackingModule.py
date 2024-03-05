import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectCon = detectCon
        self.trackCon = trackCon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mphands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo = 0, draw = True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myhand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmlist.append([id, cx, cy])
                # for locating one of the point of the hand
                # if id == 4:
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lmlist

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0 

    detector = handDetector()

    while True:
        fr, img = cap.read()

        #important step 
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        # if len(lmlist) != 0:
        #     print(lmlist[4])

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("Video", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
