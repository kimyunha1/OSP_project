import winsound
import pyttsx3   # 텍스트를 음성으로 읽어주는 라이브러리
import threading # 소리 재생 중 화면과 카운터가 멈추는 현상을 막기 위한 라이브러리

# 연속 프레임 카운터 변수 (지속 시간 체크용)
sleep_counter = 0
phone_counter = 0
yawn_counter = 0  
# 소리가 무한 중복으로 겹쳐서 프로그램이 튕기는 것을 막기 위한 주기 변수
audio_cooldown = 0
tts_cooldown = 0 
prev_stage = 0  

# 음성 안내(TTS)를 화면 멈춤 없이 백그라운드에서 실행하는 함수
def speak_async(text):
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=_speak, daemon=True).start()

# 윈도우 비프음을 백그라운드에서 실행하는 함수
def beep_async(frequency, duration):
    threading.Thread(target=winsound.Beep, args=(frequency, duration), daemon=True).start()

# 윈도우 시스템 알림음을 백그라운드에서 실행하는 함수
def message_beep_async(type):
    threading.Thread(target=winsound.MessageBeep, args=(type,), daemon=True).start()

# 지속 시간 카운트 계산
def update_counters(best_class):
    global sleep_counter, phone_counter, yawn_counter
    if best_class == "sleep":
        sleep_counter += 1
    else:
        sleep_counter = max(0, sleep_counter - 8)

    if best_class == "phone":
        phone_counter += 1
    else:
        phone_counter = max(0, phone_counter - 8)
        
    if best_class == "yawn": 
        yawn_counter += 1
    else:
        yawn_counter = max(0, yawn_counter - 2)

# 최대 누적 위험 수치 선택
def get_stage():
    current_hazard_score = max(sleep_counter, phone_counter)

    if current_hazard_score >= 75:
        return 3, current_hazard_score
    elif current_hazard_score >= 30:
        return 2, current_hazard_score
    elif current_hazard_score >= 10 or yawn_counter >= 8:
        return 1, current_hazard_score
    else:
        return 0, current_hazard_score

# -------------------------------------------------------------
# 위험도 단계별 조건문 및 백그라운드 사운드 연동 (딜레이 완전 제거)
# -------------------------------------------------------------

def trigger_alert(current_stage, best_class):
    global audio_cooldown, tts_cooldown, prev_stage

    if current_stage > prev_stage:
        audio_cooldown = 0
        tts_cooldown = 0
    prev_stage = current_stage

    text_to_show = "[상태] 정상 운전 중"
    text_color = (0, 255, 0)

    # 3단계: 초고위험 상태 판단 시 즉각적이고 강한 사이렌 경고음 및 위험 알림 음성 출력
    # 위험 행동이 약 2.5초 이상 장시간 지속될 때 발생
    if current_stage == 3:
        text_to_show = "3단계: [위험] 초고위험 상태! 즉시 운전에 집중하십시오."
        text_color = (0, 0, 255)
        if audio_cooldown == 0:
            beep_async(2500, 200)
            audio_cooldown = 8
        if tts_cooldown == 0:
            speak_async("위험 상태입니다. 즉시 행동을 중단하고 전방을 주시하십시오.")
            tts_cooldown = 90
    
    # 2단계: 음성 알림 안내 (상태가 지속될 경우 운전자 주의 유도)
    # 위험 행동이 약 1초~2.5초 사이로 지속되는 단계
    elif current_stage == 2:
        text_to_show = "2단계: [경고] 위험 행동 지속 - 음성 안내 방송 중"
        text_color = (0, 69, 255)
        if audio_cooldown == 0:
            beep_async(1500, 200)
            audio_cooldown = 12
        if tts_cooldown == 0:
            if best_class == "sleep":
                speak_async("졸음이 감지되었습니다. 전방을 주시하십시오.")
            elif best_class == "phone":
                speak_async("운전 중 휴대폰 사용은 위험합니다.")
            tts_cooldown = 90

    # 1단계: 화면 알림 및 간단한 알림 소리 
    # 위험 행동의 전조 혹은 짧은 발생 시점 (약 0.3초~1초)
    elif current_stage == 1:
        _, current_hazard_score = get_stage()
        if yawn_counter >= 8 and current_hazard_score < 10:
            text_to_show = "1단계: [주의] 하품 감지 - 화면 알림 수신"
        else:
            text_to_show = "1단계: [알림] 위험 행동 감지 - 화면 경고 메시지"
        text_color = (0, 165, 255)
        if audio_cooldown == 0:
            beep_async(800, 80)
            audio_cooldown = 5
    # 오디오 출력 오버플로우 방지용 쿨타임 카운터 다운
    if audio_cooldown > 0:
        audio_cooldown -= 1
    if tts_cooldown > 0:
        tts_cooldown -= 1

    return text_to_show, text_color