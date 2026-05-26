import gym
from gym import spaces
import numpy as np
import sys
import snakeoil3_gym as snakeoil3
import copy
import collections as col
import os
import time
import subprocess

class TorcsEnv:

    terminal_judge_start = 500
    termination_limit_progress = 5
    default_speed = 50

    initial_reset = True

    def __init__(self, vision=False, throttle=False, gear_change=False):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.initial_run = True

        self.client = snakeoil3.Client(p=3001, vision=self.vision)
        self.client.MAX_STEPS = np.inf
        self.client.get_servers_input()
        
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
        client = self.client
        this_action = self.agent_to_torcs(u)
        action_torcs = client.R.d
        action_torcs['steer'] = this_action['steer']

        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01
            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2
            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action['brake']

        if self.gear_change is True:
            action_torcs['gear'] = int(round(this_action['gear']))
        else:
            action_torcs['gear'] = 1

        obs_pre = copy.deepcopy(client.S.d)
        client.respond_to_server()
        client.get_servers_input()
        obs = client.S.d
        self.observation = self.make_observaton(obs)

        # Basic reward for gym compatibility, but reinforce_optimization uses custom_reward
        sp = np.array(obs['speedX'])
        progress = sp * np.cos(obs['angle'])
        reward = progress

        episode_terminate = False
        # Ignoriamo l'uscita pista per i primi 50 step per permettere alla fisica di stabilizzarsi dopo il reset
        if self.time_step > 50 and abs(obs['trackPos']) > 2.1:
            episode_terminate = True
            client.R.d['meta'] = True

        if self.terminal_judge_start < self.time_step:
            if sp < self.termination_limit_progress:
                episode_terminate = True
                client.R.d['meta'] = True

        if client.R.d['meta'] is True:
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1
        return self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        self.time_step = 0
        
        if self.initial_reset is True:
            self.initial_reset = False
            self.observation = self.make_observaton(self.client.S.d)
            return self.get_obs()

        # --- SOFT RESET STRATEGY ---
        self.client.R.d['meta'] = 1
        self.client.respond_to_server()
        self.client.shutdown()
        
        if relaunch is True:
            # Solo se espressamente richiesto (es. crash totale) facciamo il reset pesante
            self.reset_torcs()
        else:
            # Sequenza rapida per saltare i menu senza chiudere il gioco
            print(">>> Soft Reset: Invio sequenza ENTER (Timing Robusto)...")
            ps_cmd = "$wshell = New-Object -ComObject WScript.Shell; " + \
                     "$wshell.AppActivate('TORCS'); " + \
                     "Sleep 0.8; $wshell.SendKeys('{ENTER}'); " + \
                     "Sleep 0.8; $wshell.SendKeys('{ENTER}'); " + \
                     "Sleep 0.8; $wshell.SendKeys('{ENTER}')"
            subprocess.Popen(['powershell', '-Command', ps_cmd], shell=True)
            time.sleep(4.5) # Tempo tecnico sicuro per il caricamento fisica
            
        # Riconnessione
        self.client = snakeoil3.Client(p=3001, vision=self.vision)
        self.client.MAX_STEPS = np.inf
        self.client.get_servers_input()
        self.client.R.d['meta'] = 0
        
        self.observation = self.make_observaton(self.client.S.d)
        return self.get_obs()

    def end(self):
        if sys.platform == 'win32':
            os.system('taskkill /F /IM wtorcs.exe /T > NUL 2>&1')
        else:
            os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
        if sys.platform == 'win32':
            os.system('taskkill /F /IM wtorcs.exe /T > NUL 2>&1')
        else:
            os.system('pkill torcs')
        time.sleep(1.5)
        
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        torcs_dir = os.path.abspath(os.path.join(current_file_dir, '..', 'torcs'))
        torcs_exe = os.path.join(torcs_dir, 'wtorcs.exe')

        if sys.platform == 'win32':
            subprocess.Popen([torcs_exe, '-nofuel', '-nodamage', '-nolaptime', '-nofig'], cwd=torcs_dir, shell=True)
            print(">>> Tentativo di Autostart ROBUSTO (Nuclear Restart)...")
            # Strategia: Redondanza di AppActivate e timing calibrato per caricamento fisica
            ps_cmd = "$wshell = New-Object -ComObject WScript.Shell; " + \
                     "Sleep 10; " + \
                     "for($i=0; $i -lt 3; $i++) { $wshell.AppActivate('TORCS'); Sleep 1; } " + \
                     "$wshell.SendKeys('{ENTER}'); " + \
                     "Sleep 1; $wshell.AppActivate('TORCS'); " + \
                     "Sleep 0.5; $wshell.SendKeys('{DOWN}'); " + \
                     "Sleep 0.5; $wshell.SendKeys('{DOWN}'); " + \
                     "Sleep 0.5; $wshell.SendKeys('{DOWN}'); " + \
                     "Sleep 0.5; $wshell.SendKeys('{DOWN}'); " + \
                     "Sleep 0.5; $wshell.SendKeys('{DOWN}'); " + \
                     "Sleep 1; $wshell.AppActivate('TORCS'); " + \
                     "Sleep 0.5; $wshell.SendKeys('{ENTER}'); " + \
                     "Sleep 2; $wshell.SendKeys('{ENTER}'); " + \
                     "Sleep 2; $wshell.SendKeys('{ENTER}')"
            subprocess.Popen(['powershell', '-Command', ps_cmd], shell=True)
            time.sleep(22.0)
        else:
            os.system('torcs -nofuel -nodamage -nolaptime &')
            time.sleep(2.0)
            os.system('sh autostart.sh')
            time.sleep(1.0)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}
        if self.throttle is True:
            torcs_action.update({'accel': u[1], 'brake': u[2]})
        if self.gear_change is True:
            torcs_action.update({'gear': u[3]})
        return torcs_action

    def make_observaton(self, raw_obs):
        names = ['focus', 'speedX', 'speedY', 'speedZ', 'opponents', 'rpm', 'track', 'wheelSpinVel']
        Observation = col.namedtuple('Observaion', names)
        return Observation(focus=np.array(raw_obs.get('focus', [0]*5), dtype=np.float32)/200.,
                           speedX=np.array(raw_obs.get('speedX', 0), dtype=np.float32)/self.default_speed,
                           speedY=np.array(raw_obs.get('speedY', 0), dtype=np.float32)/self.default_speed,
                           speedZ=np.array(raw_obs.get('speedZ', 0), dtype=np.float32)/self.default_speed,
                           opponents=np.array(raw_obs.get('opponents', [0]*36), dtype=np.float32)/200.,
                           rpm=np.array(raw_obs.get('rpm', 0), dtype=np.float32),
                           track=np.array(raw_obs.get('track', [0]*19), dtype=np.float32)/200.,
                           wheelSpinVel=np.array(raw_obs.get('wheelSpinVel', [0]*4), dtype=np.float32))
