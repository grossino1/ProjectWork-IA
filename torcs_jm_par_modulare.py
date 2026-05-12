import socket
import sys
import getopt
import os
import time
PI= 3.14159265359

data_size = 2**17

ophelp=  'Options:\n'
ophelp+= ' --host, -H <host>    TORCS server host. [localhost]\n'
ophelp+= ' --port, -p <port>    TORCS port. [3001]\n'
ophelp+= ' --id, -i <id>        ID for server. [SCR]\n'
ophelp+= ' --steps, -m <#>      Maximum simulation steps. 1 sec ~ 50 steps. [100000]\n'
ophelp+= ' --episodes, -e <#>   Maximum learning episodes. [1]\n'
ophelp+= ' --track, -t <track>  Your name for this track. Used for learning. [unknown]\n'
ophelp+= ' --stage, -s <#>      0=warm up, 1=qualifying, 2=race, 3=unknown. [3]\n'
ophelp+= ' --debug, -d          Output full telemetry.\n'
ophelp+= ' --help, -h           Show this help.\n'
ophelp+= ' --version, -v        Show current version.'
usage= 'Usage: %s [ophelp [optargs]] \n' % sys.argv[0]
usage= usage + ophelp
version= "20130505-2"

def clip(v,lo,hi):
    if v<lo: return lo
    elif v>hi: return hi
    else: return v

def bargraph(x,mn,mx,w,c='X'):
    '''Draws a simple asciiart bar graph. Very handy for
    visualizing what's going on with the data.
    x= Value from sensor, mn= minimum plottable value,
    mx= maximum plottable value, w= width of plot in chars,
    c= the character to plot with.'''
    if not w: return '' # No width!
    if x<mn: x= mn      # Clip to bounds.
    if x>mx: x= mx      # Clip to bounds.
    tx= mx-mn # Total real units possible to show on graph.
    if tx<=0: return 'backwards' # Stupid bounds.
    upw= tx/float(w) # X Units per output char width.
    if upw<=0: return 'what?' # Don't let this happen.
    negpu, pospu, negnonpu, posnonpu= 0,0,0,0
    if mn < 0: # Then there is a negative part to graph.
        if x < 0: # And the plot is on the negative side.
            negpu= -x + min(0,mx)
            negnonpu= -mn + x
        else: # Plot is on pos. Neg side is empty.
            negnonpu= -mn + min(0,mx) # But still show some empty neg.
    if mx > 0: # There is a positive part to the graph
        if x > 0: # And the plot is on the positive side.
            pospu= x - max(0,mn)
            posnonpu= mx - x
        else: # Plot is on neg. Pos side is empty.
            posnonpu= mx - max(0,mn) # But still show some empty pos.
    nnc= int(negnonpu/upw)*'-'
    npc= int(negpu/upw)*c
    ppc= int(pospu/upw)*c
    pnc= int(posnonpu/upw)*'_'
    return '[%s]' % (nnc+npc+ppc+pnc)

class Client():
    def __init__(self,H=None,p=None,i=None,e=None,t=None,s=None,d=None,vision=False):
        self.vision = vision

        self.host= 'localhost'
        self.port= 3001
        self.sid= 'SCR'
        self.maxEpisodes=1 # "Maximum number of learning episodes to perform"
        self.trackname= 'unknown'
        self.stage= 3 # 0=Warm-up, 1=Qualifying 2=Race, 3=unknown <Default=3>
        self.debug= False
        self.maxSteps= 100000  # 50steps/second
        self.parse_the_command_line()
        if H: self.host= H
        if p: self.port= p
        if i: self.sid= i
        if e: self.maxEpisodes= e
        if t: self.trackname= t
        if s: self.stage= s
        if d: self.debug= d
        self.S= ServerState()
        self.R= DriverAction()
        self.setup_connection()

    def setup_connection(self):
        try:
            self.so= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as emsg:
            print('Error: Could not create socket...')
            sys.exit(-1)
        self.so.settimeout(1)

        n_fail = 5
        while True:
            a= "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"

            initmsg='%s(init %s)' % (self.sid,a)

            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error as emsg:
                sys.exit(-1)
            sockdata= str()
            try:
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print("Waiting for server on %d............" % self.port)
                print("Count Down : " + str(n_fail))
                if n_fail < 0:
                    print("relaunch torcs")
                    os.system('pkill torcs')
                    time.sleep(1.0)
                    if self.vision is False:
                        os.system('torcs -nofuel -nodamage -nolaptime &')
                    else:
                        os.system('torcs -nofuel -nodamage -nolaptime -vision &')

                    time.sleep(1.0)
                    os.system('sh autostart.sh')
                    n_fail = 5
                n_fail -= 1

            identify = '***identified***'
            if identify in sockdata:
                print("Client connected on %d.............." % self.port)
                break

    def parse_the_command_line(self):
        try:
            (opts, args) = getopt.getopt(sys.argv[1:], 'H:p:i:m:e:t:s:dhv',
                       ['host=','port=','id=','steps=',
                        'episodes=','track=','stage=',
                        'debug','help','version'])
        except getopt.error as why:
            print('getopt error: %s\n%s' % (why, usage))
            sys.exit(-1)
        try:
            for opt in opts:
                if opt[0] == '-h' or opt[0] == '--help':
                    print(usage)
                    sys.exit(0)
                if opt[0] == '-d' or opt[0] == '--debug':
                    self.debug= True
                if opt[0] == '-H' or opt[0] == '--host':
                    self.host= opt[1]
                if opt[0] == '-i' or opt[0] == '--id':
                    self.sid= opt[1]
                if opt[0] == '-t' or opt[0] == '--track':
                    self.trackname= opt[1]
                if opt[0] == '-s' or opt[0] == '--stage':
                    self.stage= int(opt[1])
                if opt[0] == '-p' or opt[0] == '--port':
                    self.port= int(opt[1])
                if opt[0] == '-e' or opt[0] == '--episodes':
                    self.maxEpisodes= int(opt[1])
                if opt[0] == '-m' or opt[0] == '--steps':
                    self.maxSteps= int(opt[1])
                if opt[0] == '-v' or opt[0] == '--version':
                    print('%s %s' % (sys.argv[0], version))
                    sys.exit(0)
        except ValueError as why:
            print('Bad parameter \'%s\' for option %s: %s\n%s' % (
                                       opt[1], opt[0], why, usage))
            sys.exit(-1)
        if len(args) > 0:
            print('Superflous input? %s\n%s' % (', '.join(args), usage))
            sys.exit(-1)

    def get_servers_input(self):
        '''Server's input is stored in a ServerState object'''
        if not self.so: return
        sockdata= str()

        while True:
            try:
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print('.', end=' ')
            if '***identified***' in sockdata:
                print("Client connected on %d.............." % self.port)
                continue
            elif '***shutdown***' in sockdata:
                print((("Server has stopped the race on %d. "+
                        "You were in %d place.") %
                        (self.port,self.S.d['racePos'])))
                self.shutdown()
                return
            elif '***restart***' in sockdata:
                print("Server has restarted the race on %d." % self.port)
                self.shutdown()
                return
            elif not sockdata: # Empty?
                continue       # Try again.
            else:
                self.S.parse_server_str(sockdata)
                if self.debug:
                    sys.stderr.write("\x1b[2J\x1b[H") # Clear for steady output.
                    print(self.S)
                break # Can now return from this function.

    def respond_to_server(self):
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s Message %s" % (emsg[1],str(emsg[0])))
            sys.exit(-1)
        if self.debug: print(self.R.fancyout())

    def shutdown(self):
        if not self.so: return
        print(("Race terminated or %d steps elapsed. Shutting down %d."
               % (self.maxSteps,self.port)))
        self.so.close()
        self.so = None

class ServerState():
    '''What the server is reporting right now.'''
    def __init__(self):
        self.servstr= str()
        self.d= dict()

    def parse_server_str(self, server_string):
        '''Parse the server string.'''
        self.servstr= server_string.strip()[:-1]
        sslisted= self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w= i.split(' ')
            self.d[w[0]]= destringify(w[1:])

    def __repr__(self):
        return self.fancyout()
        out= str()
        for k in sorted(self.d):
            strout= str(self.d[k])
            if type(self.d[k]) is list:
                strlist= [str(i) for i in self.d[k]]
                strout= ', '.join(strlist)
            out+= "%s: %s\n" % (k,strout)
        return out

    def fancyout(self):
        '''Specialty output for useful ServerState monitoring.'''
        out= str()
        sensors= [ # Select the ones you want in the order you want them.
        'stucktimer',
        'fuel',
        'distRaced',
        'distFromStart',
        'opponents',
        'wheelSpinVel',
        'z',
        'speedZ',
        'speedY',
        'speedX',
        'targetSpeed',
        'rpm',
        'skid',
        'slip',
        'track',
        'trackPos',
        'angle',
        ]

        for k in sensors:
            if type(self.d.get(k)) is list: # Handle list type data.
                if k == 'track': # Nice display for track sensors.
                    strout= str()
                    raw_tsens= ['%.1f'%x for x in self.d['track']]
                    strout+= ' '.join(raw_tsens[:9])+'_'+raw_tsens[9]+'_'+' '.join(raw_tsens[10:])
                elif k == 'opponents': # Nice display for opponent sensors.
                    strout= str()
                    for osensor in self.d['opponents']:
                        if   osensor >190: oc= '_'
                        elif osensor > 90: oc= '.'
                        elif osensor > 39: oc= chr(int(osensor/2)+97-19)
                        elif osensor > 13: oc= chr(int(osensor)+65-13)
                        elif osensor >  3: oc= chr(int(osensor)+48-3)
                        else: oc= '?'
                        strout+= oc
                    strout= ' -> '+strout[:18] + ' ' + strout[18:]+' <-'
                else:
                    strlist= [str(i) for i in self.d[k]]
                    strout= ', '.join(strlist)
            else: # Not a list type of value.
                if k == 'gear': # This is redundant now since it's part of RPM.
                    gs= '_._._._._._._._._'
                    p= int(self.d['gear']) * 2 + 2  # Position
                    l= '%d'%self.d['gear'] # Label
                    if l=='-1': l= 'R'
                    if l=='0':  l= 'N'
                    strout= gs[:p]+ '(%s)'%l + gs[p+3:]
                elif k == 'damage':
                    strout= '%6.0f %s' % (self.d[k], bargraph(self.d[k],0,10000,50,'~'))
                elif k == 'fuel':
                    strout= '%6.0f %s' % (self.d[k], bargraph(self.d[k],0,100,50,'f'))
                elif k == 'speedX':
                    cx= 'X'
                    if self.d[k]<0: cx= 'R'
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k],-30,300,50,cx))
                elif k == 'speedY': # This gets reversed for display to make sense.
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k]*-1,-25,25,50,'Y'))
                elif k == 'speedZ':
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k],-13,13,50,'Z'))
                elif k == 'z':
                    strout= '%6.3f %s' % (self.d[k], bargraph(self.d[k],.3,.5,50,'z'))
                elif k == 'trackPos': # This gets reversed for display to make sense.
                    cx='<'
                    if self.d[k]<0: cx= '>'
                    strout= '%6.3f %s' % (self.d[k], bargraph(self.d[k]*-1,-1,1,50,cx))
                elif k == 'stucktimer':
                    if self.d[k]:
                        strout= '%3d %s' % (self.d[k], bargraph(self.d[k],0,300,50,"'"))
                    else: strout= 'Not stuck!'
                elif k == 'rpm':
                    g= self.d['gear']
                    if g < 0:
                        g= 'R'
                    else:
                        g= '%1d'% g
                    strout= bargraph(self.d[k],0,10000,50,g)
                elif k == 'angle':
                    asyms= [
                          "  !  ", ".|'  ", "./'  ", "_.-  ", ".--  ", "..-  ",
                          "---  ", ".__  ", "-._  ", "'-.  ", "'\.  ", "'|.  ",
                          "  |  ", "  .|'", "  ./'", "  .-'", "  _.-", "  __.",
                          "  ---", "  --.", "  -._", "  -..", "  '\.", "  '|."  ]
                    rad= self.d[k]
                    deg= int(rad*180/PI)
                    symno= int(.5+ (rad+PI) / (PI/12) )
                    symno= symno % (len(asyms)-1)
                    strout= '%5.2f %3d (%s)' % (rad,deg,asyms[symno])
                elif k == 'skid': # A sensible interpretation of wheel spin.
                    frontwheelradpersec= self.d['wheelSpinVel'][0]
                    skid= 0
                    if frontwheelradpersec:
                        skid= .5555555555*self.d['speedX']/frontwheelradpersec - .66124
                    strout= bargraph(skid,-.05,.4,50,'*')
                elif k == 'slip': # A sensible interpretation of wheel spin.
                    frontwheelradpersec= self.d['wheelSpinVel'][0]
                    slip= 0
                    if frontwheelradpersec:
                        slip= ((self.d['wheelSpinVel'][2]+self.d['wheelSpinVel'][3]) -
                              (self.d['wheelSpinVel'][0]+self.d['wheelSpinVel'][1]))
                    strout= bargraph(slip,-5,150,50,'@')
                else:
                    strout= str(self.d[k])
            out+= "%s: %s\n" % (k,strout)
        return out

class DriverAction():
    '''What the driver is intending to do (i.e. send to the server).
    Composes something like this for the server:
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus 0)(meta 0) or
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus -90 -45 0 45 90)(meta 0)'''
    def __init__(self):
       self.actionstr= str()
       self.d= { 'accel':0.2,
                   'brake':0,
                  'clutch':0,
                    'gear':1,
                   'steer':0,
                   'focus':[-90,-45,0,45,90],
                    'meta':0
                    }

    def clip_to_limits(self):
        """There pretty much is never a reason to send the server
        something like (steer 9483.323). This comes up all the time
        and it's probably just more sensible to always clip it than to
        worry about when to. The "clip" command is still a snakeoil
        utility function, but it should be used only for non standard
        things or non obvious limits (limit the steering to the left,
        for example). For normal limits, simply don't worry about it."""
        self.d['steer']= clip(self.d['steer'], -1, 1)
        self.d['brake']= clip(self.d['brake'], 0, 1)
        self.d['accel']= clip(self.d['accel'], 0, 1)
        self.d['clutch']= clip(self.d['clutch'], 0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear']= 0
        if self.d['meta'] not in [0,1]:
            self.d['meta']= 0
        if type(self.d['focus']) is not list or min(self.d['focus'])<-180 or max(self.d['focus'])>180:
            self.d['focus']= 0

    def __repr__(self):
        self.clip_to_limits()
        out= str()
        for k in self.d:
            out+= '('+k+' '
            v= self.d[k]
            if not type(v) is list:
                out+= '%.3f' % v
            else:
                out+= ' '.join([str(x) for x in v])
            out+= ')'
        return out
        return out+'\n'

    def fancyout(self):
        '''Specialty output for useful monitoring of bot's effectors.'''
        out= str()
        od= self.d.copy()
        od.pop('gear','') # Not interesting.
        od.pop('meta','') # Not interesting.
        od.pop('focus','') # Not interesting. Yet.
        for k in sorted(od):
            if k == 'clutch' or k == 'brake' or k == 'accel':
                strout=''
                strout= '%6.3f %s' % (od[k], bargraph(od[k],0,1,50,k[0].upper()))
            elif k == 'steer': # Reverse the graph to make sense.
                strout= '%6.3f %s' % (od[k], bargraph(od[k]*-1,-1,1,50,'S'))
            else:
                strout= str(od[k])
            out+= "%s: %s\n" % (k,strout)
        return out

def destringify(s):
    '''makes a string into a value or a list of strings into a list of
    values (if possible)'''
    if not s: return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            print("Could not find a value in %s" % s)
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]

#############################################
# MODULAR DRIVE LOGIC WITH USER PARAMETERS  #
#############################################

import math
import csv  
import time

# ================= USER CONFIGURABLE PARAMETERS =================
# Queste variabili globali sono il "Setup" della tua auto. 
# Modificarle cambia il carattere della vettura senza toccare la matematica complessa.

TARGET_SPEED = 300       # Velocità massima assoluta che l'auto cercherà di raggiungere nei lunghi rettilinei.
STEER_GAIN = 20          # Sensibilità dello sterzo: quanto bruscamente gira le ruote in base all'angolo della pista.
CENTERING_GAIN = 0.1     # "Forza di attrazione" verso il centro. A 0.1 è debole, permettendo all'auto di allargarsi sui cordoli.
BRAKE_THRESHOLD = 0.1    # (Non utilizzato in questo blocco, ma di solito indica una soglia di attivazione del freno)
GEAR_SPEEDS = [0, 60, 95, 150, 230, 270]  # Le velocità (in km/h) a cui la macchina passa alla marcia successiva (1a, 2a, 3a, ecc.)
ENABLE_TRACTION_CONTROL = True # Interruttore per attivare/disattivare il sistema anti-pattinamento.

# ================= HELPER FUNCTIONS =================

def calculate_steering(S):
    # --- STERZO DINAMICO ---
    # Come nelle auto vere, a bassa velocità serve girare di più il volante per fare la curva.
    dynamic_gain = STEER_GAIN
    if S['speedX'] < 80:
        # Se andiamo a meno di 80 km/h, aumentiamo la sensibilità dello sterzo del 50%.
        # Questo aiuta tantissimo nei tornanti stretti come il Cavatappi.
        dynamic_gain = STEER_GAIN * 1.5

    # Formula dello sterzo:
    # 1. (S['angle'] * dynamic_gain) -> Segue l'angolo della pista.
    # 2. - (S['trackPos'] * CENTERING_GAIN) -> Piccola correzione per non finire sull'erba.
    steer = (S['angle'] * dynamic_gain / math.pi) - (S['trackPos'] * CENTERING_GAIN)
    
    # Assicuriamoci che il valore inviato al server sia tra -1.0 (tutto a destra) e 1.0 (tutto a sinistra).
    return max(-1.0, min(1.0, steer))

def calculate_speed_logic(S):
    speed = S['speedX']
    
    # --- PERCEZIONE VISIVA ---
    # Invece di guardare solo un punto davanti (sensore 9), guardiamo un "ventaglio" di 5 sensori centrali.
    # Questo evita che l'auto freni per sbaglio se sbanda un attimo e il muso punta il muro.
    front_vision = S['track'][5:14]
    max_dist_ahead = max(front_vision) # Prende la distanza libera più lunga disponibile davanti.

    # --- CALCOLO VELOCITÀ IDEALE (SAFE SPEED) ---
    # Regola base: Velocità Sicura = Distanza visibile moltiplicata per 2.
    # Esempio: Vedo a 150 metri -> Posso andare a 300 km/h. Vedo a 50 metri (curva vicina) -> Devo scendere a 100 km/h.
    safe_speed = max_dist_ahead * 2.5
    
    # Limitiamo la safe_speed per evitare comportamenti estremi:
    # - Non scende mai sotto i 65 km/h (altrimenti si pianta in curva perdendo slancio).
    # - Non supera mai la TARGET_SPEED massima (300 km/h).
    safe_speed = max(75.0, min(TARGET_SPEED, safe_speed))

    # --- CONTROLLO DEI PEDALI (PROPORZIONALE) ---
    accel = 0.0
    brake = 0.0
    diff = safe_speed - speed # Calcola lo scarto tra quanto dovremmo andare e quanto stiamo andando.

    if diff > 0:
        # SIAMO LENTI (diff è positivo) -> Dobbiamo accelerare
        # Minore è la differenza, meno gas diamo (per evitare scatti), dividendo per 20.
        accel = min(1.0, diff / 20.0) 
        
        # Spinta esplosiva: se andiamo a meno di 90 km/h, diamo il 100% di gas a prescindere.
        # Serve a schizzare fuori dalle curve lente senza esitazioni.
        if speed < 90: 
            accel = 1.0
    else:
        # SIAMO TROPPO VELOCI (diff è negativo) -> Dobbiamo frenare
        # Tolleranza: Ignoriamo piccole variazioni (fino a 10 km/h sopra il limite) per far scorrere la macchina
        # senza che tocchi continuamente i freni nei falsi allarmi.
        if abs(diff) > 10:
            brake = min(1.0, (-diff) / 35.0) # (-diff) trasforma il numero in positivo per il pedale del freno.

    # --- PANIC BRAKE (SISTEMA DI EMERGENZA) ---
    # Se il sensore esattamente dritto (il 9) vede un muro a meno di 45 metri e stiamo andando forte (>80km/h),
    # ignora tutti i calcoli dolci e INCHIODA (freno al 100%).
    if S['track'][9] < 45 and speed > 80:
        brake = 1.0

    return accel, brake

def shift_gears(S):
    # --- CAMBIO MARCE ---
    gear = 1
    speed = S['speedX']
    # Controlla la nostra velocità contro la lista GEAR_SPEEDS definita in alto.
    # Scala le marce dinamicamente se la velocità scende (aiuta anche il freno motore).
    for i, threshold in enumerate(GEAR_SPEEDS):
        if speed > threshold:
            gear = i + 1
    return gear

def traction_control(S, accel):
    # --- CONTROLLO TRAZIONE (TCS) ---
    if ENABLE_TRACTION_CONTROL:
        # Calcola quanto slittano le ruote (Rotazione ruote posteriori meno rotazione ruote anteriori).
        slip = (S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) - (S['wheelSpinVel'][0] + S['wheelSpinVel'][1])
        
        # Se le ruote dietro girano molto più a vuoto di quelle davanti (>5)...
        if slip > 5:
            # ...taglia la potenza del motore (-0.3) per far riprendere aderenza alle gomme.
            accel -= 0.3
            
    # Assicura che l'acceleratore non diventi mai un numero negativo.
    return max(0.0, accel)

# ================= MAIN DRIVE FUNCTION =================
def drive_modular(c):
    # Estrae i Sensori (S) e i Comandi da inviare (R)
    S, R = c.S.d, c.R.d
    
    # 1. Chiede alla funzione helper quanto girare il volante.
    R['steer'] = calculate_steering(S)
    
    # 2. Chiede alla funzione helper quanto gas o freno dare.
    accel, brake = calculate_speed_logic(S)
    
    # --- 3. TRAIL BRAKING FISICO ---
    # Concetto avanzato di corsa: se sterzo molto, devo frenare di meno, altrimenti le gomme anteriori 
    # si bloccano (sottosterzo) e la macchina va dritta.
    # Qui sottraiamo fino al 50% della pressione del freno in base a quanto è girato il volante.
    brake = brake * (1.0 - (abs(R['steer']) * 0.5))

    # 4. Applica l'acceleratore, passandolo prima attraverso il filtro anti-pattinamento (TCS).
    R['accel'] = traction_control(S, accel)
    
    # 5. Applica il freno definitivo (assicurandosi sia tra 0 e 100%).
    R['brake'] = max(0.0, min(1.0, brake))
    
    # 6. Cambia la marcia.
    R['gear'] = shift_gears(S)
    
    # --- 7. SICUREZZA FINALE ---
    # In fisica non si frena e accelera contemporaneamente (nella guida normale).
    # Se stiamo toccando il freno, azzera l'acceleratore.
    if R['brake'] > 0.05:
        R['accel'] = 0.0
        
    return

# ================= MAIN LOOP =================
# Questo è il motore del programma. Continua a girare all'infinito (fino al massimo degli step).
# ================= MAIN LOOP CON DATA LOGGER OTTIMIZZATO =================
if __name__ == "__main__":
    C = Client(p=3001) 
    
    print("=========================================")
    print("BOT 1:49 AVVIATO - DATASET OTTIMIZZATO")
    print("=========================================")
    
    dataset_filename = "dataset_bot_154_clean.csv"
    csv_file = open(dataset_filename, "w", newline='')
    csv_writer = csv.writer(csv_file)
    
    # --- LA NOSTRA BLACKLIST ---
    # Inseriamo qui tutto il "rumore" che non serve all'IA per imparare a guidare da sola
    KEYS_TO_IGNORE = [
        'opponents', 'focus', 'fuel', 'damage', 'z', 
        'curLapTime', 'lastLapTime', 'distFromStart', 'distRaced', 'racePos'
    ]
    
    headers_written = False
    step_count = 0
    t0 = time.time()
    
    try:
        for step in range(C.maxSteps, 0, -1):
            C.get_servers_input()  
            drive_modular(C)       
            
            # --- SEZIONE DI LOGGING INTELLIGENTE ---
            if not headers_written:
                headers = ["timestamp", "target_steer", "target_accel", "target_brake", "target_gear"]
                for key, value in sorted(C.S.d.items()):
                    if key in KEYS_TO_IGNORE: # Se la chiave è nella blacklist, saltala!
                        continue
                        
                    if isinstance(value, list):
                        for i in range(len(value)):
                            headers.append(f"{key}_{i}")
                    else:
                        headers.append(key)
                csv_writer.writerow(headers)
                headers_written = True

            current_time = time.time() - t0
            row = [current_time, C.R.d['steer'], C.R.d['accel'], C.R.d['brake'], C.R.d['gear']]
            
            for key in sorted(C.S.d.keys()):
                if key in KEYS_TO_IGNORE: # Saltiamo i dati inutili anche qui
                    continue
                    
                val = C.S.d[key]
                if isinstance(val, list):
                    row.extend(val)
                else:
                    row.append(val)
                    
            csv_writer.writerow(row)
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"Registrati {step_count} step puliti... (Vel: {int(C.S.d['speedX'])} km/h)")

            C.respond_to_server()  
            
    except KeyboardInterrupt:
        print("\nRegistrazione interrotta manualmente.")
        
    finally:
        csv_file.close()
        C.shutdown()
        print("=========================================")
        print(f"Salvataggio completato! File: {dataset_filename}")
        print(f"Righe totali: {step_count}")
        print("=========================================")