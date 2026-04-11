#!/bin/bash

# ROS beállítása a szervizek lekérdezéséhez
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source install/setup.bash

# Beállítások
TIMEOUT_LIMIT=240       # Hány másodperc elérhetetlenség után lője ki (240 mp = 4 perc)
CHECK_INTERVAL=10       # Hány másodpercenként csekkolja a rendszert

echo "*********************************************************"
echo "* FELÜGYELT TRÉNING (WATCHDOG) ELINDÍTVA                *"
echo "* Automatikus újraindítás fagyás vagy leállás esetén. *"
echo "*********************************************************"

cleanup_and_exit() {
    echo ""
    echo "====================================================="
    echo "[WATCHDOG] Kézi leállítás (Ctrl+C) észlelve!"
    echo "[WATCHDOG] Minden folyamat végleges leállítása és kilépés..."
    echo "====================================================="
    kill -9 $MAIN_PID 2>/dev/null
    pkill -9 -f train.py 2>/dev/null
    pkill -9 -f gzserver 2>/dev/null
    pkill -9 -f gzclient 2>/dev/null
    exit 0
}

# A Ctrl+C (SIGINT) és SIGTERM jelek elfogása, hogy ne induljon újra
trap cleanup_and_exit SIGINT SIGTERM

while true; do
    echo "====================================================="
    echo "[WATCHDOG] Új tréning munkamenet indítása..."
    echo "====================================================="
    
    rm -f /tmp/gazebo_fatal_error.flag
    
    # A tréning szkript elindítása a háttérben
    bash ./start_training.sh &
    MAIN_PID=$!
    
    # Indulás után adunk neki egy kis időt, amin belül normális, hogy még nincs service
    echo "[WATCHDOG] Várakozás a rendszerek betöltésére (45 másodperc)..."
    sleep 45
    
    HANG_TIMER=0
    
    # Belső ciklus, ami addig fut, amíg a start_training.sh él a háttérben
    while kill -0 $MAIN_PID 2>/dev/null; do
        
        # Gyorsabb ellenőrzés a Python szkript felől küldött FATAL hibáról
        if [ -f "/tmp/gazebo_fatal_error.flag" ]; then
            echo "====================================================="
            echo "[WATCHDOG] Olyan jelzést kaptunk a Python kód felől, hogy a robot respawn"
            echo "           több próbálkozás (attempt) alapján sem sikerült!"
            echo "[WATCHDOG] Folyamatok azonnali lelövése és újraindítása..."
            echo "====================================================="
            rm -f /tmp/gazebo_fatal_error.flag
            kill -9 $MAIN_PID 2>/dev/null
            pkill -9 -f train.py 2>/dev/null
            pkill -9 -f gzserver 2>/dev/null
            pkill -9 -f gzclient 2>/dev/null
            sleep 5
            break
        fi
        
        # Lekérdezzük a /spawn_entity szervizt egy 10 másodperces időkorláttal
        # Ha a Gazebo kifagy, ez a hívás beáll vagy hibát dob
        if timeout 10 ros2 service type /spawn_entity > /dev/null 2>&1; then
            HANG_TIMER=0  # Minden rendben, a számláló nullázódik
        else
            HANG_TIMER=$((HANG_TIMER + CHECK_INTERVAL))
            echo "[WATCHDOG] FIGYELEM: A szimulátor nem válaszol $HANG_TIMER másodperce..."
        fi
        
        # Ha a kifagyás meghaladta a megengedett 4 percet
        if [ $HANG_TIMER -ge $TIMEOUT_LIMIT ]; then
            echo "====================================================="
            echo "[WATCHDOG] KRITIKUS: A szimuláció végleg lefagyott (>$TIMEOUT_LIMIT mp)!"
            echo "[WATCHDOG] Folyamatok azonnali lelövése és újraindítása..."
            echo "====================================================="
            
            kill -9 $MAIN_PID 2>/dev/null
            pkill -9 -f train.py 2>/dev/null
            pkill -9 -f gzserver 2>/dev/null
            pkill -9 -f gzclient 2>/dev/null
            sleep 5
            break # A belső ciklust megszakítja, így a külső while true elölről elindítja az egészet
        fi
        
        sleep $CHECK_INTERVAL
    done
    
    echo "[WATCHDOG] Munkamenet leállt. Biztonsági takarítás az újraindítás előtt..."
    pkill -9 -f train.py 2>/dev/null
    pkill -9 -f gzserver 2>/dev/null
    pkill -9 -f gzclient 2>/dev/null
    sleep 3
    echo "[WATCHDOG] Újraindítás 3 másodperc múlva..."
    sleep 3
done
