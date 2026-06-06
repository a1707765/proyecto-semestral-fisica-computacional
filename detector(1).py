#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
  muón → atmósfera (Bethe-Bloch) → geometría detector (TOP+BOTTOM)
  → deposición Landau → scintillación → PE (QE, TTS, ganancia)
  → waveforms físicas → electrónica → ADC → trigger coincidencia
  → ToF → reconstrucción angular → histogramas

UNIDADES: MeV, cm, g/cm³, ns, V
=============================================================================
"""
'''

import numpy as np
import matplotlib as plt
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import moyal
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT = 29.9792458   # cm/ns
M_MUON  = 105.658      # MeV/c²
M_ELEC  = 0.511        # MeV/c²
K_BB    = 0.307075     # MeV·cm²/g
RNG     = np.random.default_rng() # generador de numeros aleatorios con semilla


# =============================================================================
# GENERACIÓN DE MUONES
# =============================================================================

class GeneradorDeMuones:
   

    def __init__(self,
                 alpha:     float = 2.7,
                 E_min_MeV: float = 5000,   # 5 GeV
                 E_max_MeV: float = 1.0e6):   # 1 TeV
        self.alpha   = alpha
        self.E_min   = E_min_MeV
        self.E_max   = E_max_MeV
        self._gam    = alpha - 1
        self._ratio  = (E_min_MeV / E_max_MeV) ** self._gam
        

    def AnguloCenital(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        r         = RNG.uniform(0, 1, n)
        cos_theta = np.clip((r) ** (1/3), 0, 1)
        theta     = np.arccos(cos_theta)
        phi       = RNG.uniform(0, 2 * np.pi, n)
        
        return theta, phi

    def Energia(self, n: int) -> np.ndarray:
        r = RNG.uniform(0.0, 1.0, n)
        
        return self.E_min * (1.0 - r * (1.0 - self._ratio)) ** (-1.0 / self._gam)

    def generar(self, n: int) -> List[dict]:
        theta, phi = self.AnguloCenital(n)
        E = self.Energia(n)
        
        # dirección: z positivo = descendente (hacia el detector)
        dx = np.sin(theta) * np.cos(phi)
        dy = np.sin(theta) * np.sin(phi)
        dz = -np.cos(theta)
        return [{'theta': theta[i], 'phi': phi[i], 'E0': E[i],
                 'dir': np.array([dx[i], dy[i], dz[i]])} for i in range(n)]
    


# =============================================================================
#  BETHE-BLOCH
# =============================================================================

class BetheBloch:
    

    def __init__(self, Z: float, A: float, I_eV: float, rho: float):
        self.Z   = Z
        self.A   = A
        self.I   = I_eV * 1e-6    # eV → MeV
        self.rho = rho
        self.ZA  = Z / A

    def cinetica(self, Ekin: float) -> Tuple[float, float, float]:
        """Retorna (β, βγ, γ)."""
        gam = (Ekin + M_MUON) / M_MUON
        bet = np.sqrt(max(1.0 - 1.0/gam**2, 0.0))
        return bet, bet*gam, gam

    def dEdx(self, Ekin: float) -> float:
        """MeV/cm para muón de energía cinética Ekin [MeV]."""
        if Ekin < 0.1:
            return 0.0
        beta, _, gamma = self.cinetica(Ekin)
        b2 = beta**2
        r  = M_ELEC / M_MUON
        Tmax = (2.0 * M_ELEC * b2 * gamma**2) / (1.0 + 2.0*gamma*r + r**2)
        if Tmax <= 0.0:
            return 1.5 * self.rho
        arg = 2.0 * M_ELEC * b2 * gamma**2 * Tmax / self.I**2
        if arg <= 1.0:
            return 1.5 * self.rho
        val = K_BB * self.ZA / b2 * (0.5 * np.log(arg) - b2)
        return max(val, 1.5) * self.rho   # MeV/cm


# =============================================================================
#  PROPAGACIÓN ATMOSFÉRICA
# =============================================================================

class propagacion:
    '''
    Integra Bethe-Bloch en aire a lo largo de L = H/cos(theta).
    '''

    def __init__(self,
                 H_cm:    float = 10e5,    # 10 km
                 rho_air: float = 1.225e-3,
                 n_steps: int   = 500):
        # Aire: Z_eff=7.22, A_eff=14.5, I=85.7 eV
        
        self.bb     = BetheBloch(Z=7.22, A=14.5, I_eV=85.7, rho=rho_air)
        self.H      = H_cm
        self.Nsteps = n_steps

    def propagate(self, E0_MeV: float, theta_rad: float) -> Optional[float]:
        """
        Propaga el muón desde la cima de la atmósfera.
        Retorna la energía TOTAL (cinética + masa) al llegar, o None si se detiene.
        """
        cos_t = np.cos(theta_rad) 
        L     = self.H / cos_t  # distancia real recorrida en la atmosfera
        dx    = L / self.Nsteps
        
        if cos_t <= 0:
            return None
        
        Ekin  = E0_MeV - M_MUON       # energía cinética
        
        
        for _ in range(self.Nsteps):
            if Ekin < 5.0:
                return None
            Ekin -= self.bb.dEdx(Ekin) * dx
        return Ekin + M_MUON    # energía total

##############

    def sobrevive_hasta_altura(self, E0_MeV, theta, h_cm):
        cos_t = np.cos(theta)
        if cos_t <= 0:
            return False
        
        L = h_cm / cos_t
        dx = L / self.Nsteps
        
        Ekin = E0_MeV - M_MUON
        
        for _ in range(self.Nsteps):
            if Ekin < 5.0:
                return False
            Ekin -= self.bb.dEdx(Ekin) * dx
        
        return True

# =============================================================================
# GEOMETRÍA DEL DETECTOR
# =============================================================================

@dataclass
class capa:
    """
    Capa de scintillador plástico BC-408.
    Plano horizontal en z = z_pos, activo en [cx±wx/2, cy±wy/2].
    rho = 1.032 g/cm^3, Z_eff=5.61, A_eff=11.0, I=64.7 eV.
    """
    name:      str
    z_pos:     float
    cx:        float = 0.0
    cy:        float = 0.0
    width_x:   float = 40.0
    width_y:   float = 40.0
    thickness: float = 1.0
    bb: BetheBloch = field(default_factory=lambda:
        BetheBloch(Z=5.61, A=11.0, I_eV=64.7, rho=1.032))


class Geometria:
    """
    Detector TOP (arriba) + BOTTOM (abajo), separados por `separation_cm`.

    Método de muestreo geométrico correcto:
      Se samplea un punto de paso en el PLANO MEDIO (z=0).
      Luego se extrapola hacia TOP (dirección descendente inversa)
      y hacia BOTTOM (dirección descendente).
    
    """

    def __init__(self,
                 separacion: float = 30.0,
                 ancho:   float = 80.0,
                 grosor:   float = 1.0):
        z_top = +separacion / 2.0
        z_bot = -separacion / 2.0 
        
        self.top  = capa('TOP', z_top, width_x=ancho,
                          width_y=ancho, thickness=grosor)
        
        self.bot  = capa('BOTTOM', z_bot, width_x=ancho,
                          width_y=ancho, thickness=grosor)
        
        self.sep  = separacion 

    # ── utilidades ──────────────────────────────────────────────────────

    @staticmethod
    def impacto(pos: np.ndarray, direc: np.ndarray,
                   c: capa) -> Optional[np.ndarray]:
        """
        Intersección rayo–plano z=layer.z_pos.
        direc no necesita ser unitario, sólo no nulo en z.
        Retorna punto [x,y,z] o None si no intersecta dentro del área activa.
        """ 
        dz = direc[2]
        
        if abs(dz) < 1e-12:
            return None
        
        t = (c.z_pos - pos[2]) / dz
        
        if t < 0:
            return None
        
        hit = pos + t * direc
        ok  = (abs(hit[0] - c.cx) <= c.width_x / 2.0 and
               abs(hit[1] - c.cy) <= c.width_y / 2.0)
        
        return hit if ok else None

    @staticmethod
    def track(direc: np.ndarray, thickness: float) -> float:
        """Longitud de trayectoria en capa: L = espesor / cos(θ)."""
        cos_t = abs(direc[2])
        return thickness / max(cos_t, 1e-6)

    # ── trazado principal ─────────────────────────────────────────────

    def trace(self, muon: dict) -> dict:
        """
        Traza el muón a través del detector corrigiendo los sistemas de coordenadas.
        Asegura un paso consistente por el plano medio z=0.
        """ 
        d = muon['dir'].copy()
        if d[2] < 0:
            d = -d                 # Asegurar dz positivo según el generador

        # Samplear punto de paso en z=0 dentro del área activa 
        x0 = RNG.uniform(-self.top.width_x / 2.0, self.top.width_x / 2.0)
        y0 = RNG.uniform(-self.top.width_y / 2.0, self.top.width_y / 2.0)
        p0 = np.array([x0, y0, 0.0])

        res = dict(hit_top=False, hit_bottom=False,
                   L_top=0.0,    L_bottom=0.0,
                   pos_top=None, pos_bottom=None)

        # Hacia TOP (z_top = +50): Retrocedemos en la trayectoria (invertir X, Y y forzar Z positivo)
        d_up = np.array([-d[0], -d[1], abs(d[2])]) 
        hit_top = self.impacto(p0, d_up, self.top) 
        if hit_top is not None:
            res.update(hit_top=True, pos_top=hit_top,
                       L_top=self.track(d, self.top.thickness))

        # Hacia BOTTOM (z_bot = -50): Avanzamos en la trayectoria (X, Y normales y Z negativo)
        d_dn = np.array([d[0], d[1], -abs(d[2])])
        hit_bot = self.impacto(p0, d_dn, self.bot)
        if hit_bot is not None:
            res.update(hit_bottom=True, pos_bottom=hit_bot,
                       L_bottom=self.track(d, self.bot.thickness))

        return res #diccionario


# =============================================================================
#DEPOSICIÓN DE ENERGÍA 
# =============================================================================

class Deposicion: 
   
    @staticmethod
    def sample(E_total: float, L_cm: float, c: capa) -> float:

        Ekin = E_total - M_MUON

        if Ekin < 0.5 or L_cm <= 0:
            return 0.0

        dEdx_mean = c.bb.dEdx(Ekin)   # MeV/cm

        Edep = dEdx_mean * L_cm

        return float(Edep)

# =============================================================================
#  CENTELLEO
# =============================================================================

class Centellador:
    """
    BC-408: tau = 2.1 ns, LY = 10 000 ph/MeV, eta_col = 5%.
    N_fotones ~ Poisson(E_dep · LY · eta)
    t_fotón   ~ Exp(tau)
    """

    def __init__(self,
                 LY:  float = 10_000.0,
                 tau: float = 2.1,
                 eta: float = 0.05):
        self.LY  = LY
        self.tau = tau
        self.eta = eta

    def t_foton(self, Edep: float) -> np.ndarray:
        n = RNG.poisson(max(Edep * self.LY * self.eta, 0.0))
        if n > 0:
            return RNG.exponential(self.tau, n) 
        else :
            return np.array([])


# =============================================================================
#MODELO PMT
# =============================================================================

class PMT:
    """
    QE=25%, G=10^6, sigma_G/G=40%, TTS=0.3 ns, V_PE=2.5 mV, t_transit=20 ns.

    detect() aplica QE, TTS y desplazamiento temporal t0_offset_ns.
    """

    def __init__(self,
                 QE:       float = 0.25,
                 ganancia:     float = 1.0e6,
                 gain_sig: float = 0.40, #fluctuación estadística de la ganancia
                 TTS_ns:   float = 0.30,
                 V_PE:     float = 2.5e-3,
                 transit:  float = 20.0):
        self.QE      = QE
        self.G       = ganancia
        self.Gsig    = gain_sig * ganancia
        self.TTS     = TTS_ns
        self.V_PE    = V_PE
        self.transit = transit

    def detectar(self,
               ph_times: np.ndarray,
               t0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna (t_anode, amplitudes) para los fotoelectrones generados.
        t0 es el desplazamiento físico del canal (ToF para el BOTTOM).
        """
        if len(ph_times) == 0:
            return np.array([]), np.array([])
        mask = RNG.uniform(0.0, 1.0, len(ph_times)) < self.QE
        t    = ph_times[mask]
        if len(t) == 0:
            return np.array([]), np.array([])
        t_an  = t + self.transit + RNG.normal(0.0, self.TTS, len(t)) + t0
        g     = np.abs(RNG.normal(self.G, self.Gsig, len(t)))
        amps  = self.V_PE * g / self.G
        return t_an, amps


# =============================================================================
# ELECTRÓNICA Y WAVEFORM
# =============================================================================

class Electronica:
    """
    Construye waveforms físicas sumando pulsos gaussianos de PE individuales.
    La amplitud NO se normaliza — preserva información de energía y N_PE.

    ADC: 12 bits, rango [0, 2] V, ruido gaussiano,.
   
    """

    def __init__(self,
                 t_start:    float = 0.0,
                 t_end:      float = 300.0,
                 dt:         float = 0.1,       # ns
                 noise_mV:   float = 0.5,
                 ADC_bits:   int   = 12,
                 Vrange:     float = 2.0,        # V
                 threshold_V: float = 5.0e-3,   # 5 mV
                 pulse_sig:  float = 1.8):       # ns  sigma del pulso PE
        self.t_ax     = np.arange(t_start, t_end, dt)
        self.dt       = dt
        self.noise    = noise_mV * 1e-3
        self.Vrange   = Vrange
        self.ADC_lev  = 2**ADC_bits
        self.thr      = threshold_V
        self.psig     = pulse_sig

    def forma(self,
              pe_times: np.ndarray,
              pe_amps:  np.ndarray) -> np.ndarray:
        """waveform = suma de gaussianas + ruido. Amplitud física."""
        wf = np.zeros(len(self.t_ax))
        for tp, ap in zip(pe_times, pe_amps):
            # Solo contribuyen PE dentro de la ventana + 5 sigma
            if self.t_ax[0] - 5*self.psig < tp < self.t_ax[-1] + 5*self.psig:
                wf += ap * np.exp(-0.5 * ((self.t_ax - tp) / self.psig)**2)
        wf += RNG.normal(0.0, self.noise, len(self.t_ax))
        return wf

    def digi(self, wf: np.ndarray) -> np.ndarray:
        """ADC con saturación y cuantización."""
        return (np.clip(wf, 0.0, self.Vrange) /
                self.Vrange * (self.ADC_lev - 1)).astype(np.int32)

    def threshold_tiempo(self, wf: np.ndarray) -> Optional[float]:
        """
        Primer cruce del umbral con interpolación lineal.
        Retorna tiempo (ns) o None si no supera el umbral.
        """
        above = np.where(wf >= self.thr)[0]
        if len(above) == 0:
            return None
        i = above[0]
        if i > 0 and wf[i] > wf[i-1]:
            dt_frac = (self.thr - wf[i-1]) / (wf[i] - wf[i-1])
            return self.t_ax[i-1] + dt_frac * self.dt
        return float(self.t_ax[i])

    def carga(self, wf: np.ndarray) -> float:
        return float(np.sum(np.clip(wf, 0.0, None)) * self.dt)

    def peak(self, wf: np.ndarray) -> float:
        return float(np.max(wf))


# =============================================================================
#  TRIGGER POR COINCIDENCIA
# =============================================================================

class trigger:
    """Acepta el evento si ambas capas tienen señal dentro de `window_ns`."""

    def __init__(self, window_ns: float = 5): 
        self.window = window_ns

    def aceptar(self,
               t_top: Optional[float],
               t_bot: Optional[float]) -> bool:
        if t_top is None or t_bot is None:
            return False
        return abs(t_bot - t_top) <= self.window


# =============================================================================
# RECONSTRUCCIÓN ANGULAR
# =============================================================================



class Reconstruccion:
    """
    cos(θ_rec) = d / (c · delta t)
    donde d = separación entre capas, delta t = t_bot − t_top.
   
    """

    def __init__(self, sep_cm: float):
        self.d = sep_cm

    def angle(self, dt_ns: float) -> Optional[float]:
        
        if dt_ns <= 0.0:
            return None
        
        cos_t = self.d / (C_LIGHT * dt_ns)
        
        if not (0.0 < cos_t <= 1.0):
            return None
        
        return float(np.arccos(cos_t))


# =============================================================================
# SIMULACIÓN PRINCIPAL
# =============================================================================

class MuonDetectorSim:

    def __init__(self,
                 n_muons:   int   = 30000,
                 sep_cm:    float = 30.0,  # valores por defecto 
                 width_cm:  float = 80.0,
                 thick_cm:  float = 1.0):

        self.n     = n_muons
        self.gen   = GeneradorDeMuones()
        self.atm   = propagacion()
        self.geo   = Geometria(sep_cm, width_cm, thick_cm)
        self.sci   = Centellador()
        self.pmt   = PMT()
        self.elec  = Electronica()
        self.trig  = trigger()
        self.reco  = Reconstruccion(sep_cm) 
        ############
        self.flujo_altura = {
        '1km': 0,
        '5km': 0,
        '10km': 0
    }

        self.buf = {k: [] for k in [
            'theta_true', 'theta_rec',
            'tof_true',   'tof_rec',
            'E_dep_top',  'E_dep_bot',
            'n_pe_top',   'n_pe_bot',
            'charge_top', 'charge_bot',
            'amp_top',    'amp_bot',
        ]}
        self.wf_examples: List[tuple] = []
        self.theta_generated = [] ###########################

        print("=" * 62)
        print("  SIMULACIÓN MC — DETECTOR DE MUONES CÓSMICOS")
        print("=" * 62)
        print(f"  Muones generados:   {n_muons}")
        print(f"  Separación:         {sep_cm:.0f} cm")
        print(f"  Centellador:       {width_cm}×{width_cm}×{thick_cm} cm³")
        print()

    # ── evento a evento ─────────────────────────────────────────

    def run(self) -> dict:
        muons  = self.gen.generar(self.n)
        cnts   = {'atm': 0, 'geo': 0, 'npe': 0, 'trig': 0}

        print("  Procesando eventos...")
        for i, mu in enumerate(muons):
         
            self.theta_generated.append(mu['theta'])
            
            if self.atm.sobrevive_hasta_altura(mu['E0'], mu['theta'], 1e5):
                self.flujo_altura['1km'] += 1
        
            if self.atm.sobrevive_hasta_altura(mu['E0'], mu['theta'], 5e5):
                self.flujo_altura['5km'] += 1
        
            if self.atm.sobrevive_hasta_altura(mu['E0'], mu['theta'], 1e6):
                self.flujo_altura['10km'] += 1
            if i > 0 and i % 1000 == 0:
                print(f"    {i}/{self.n}  "
                      f"triggers={cnts['trig']}  "
                      f"atm={cnts['atm']}  "
                      f"geo={cnts['geo']}  "
                      f"npe={cnts['npe']}")

            # 1. Propagación atmosférica ──────────────────────────────────
            E_arr = self.atm.propagate(mu['E0'], mu['theta'])
            if E_arr is None:
                cnts['atm'] += 1; continue

            # 2. Geometría ────────────────────────────────────────────────
            trk = self.geo.trace(mu)
            if not (trk['hit_top'] and trk['hit_bottom']):
                cnts['geo'] += 1; continue

            # 3. ToF verdadero ─────────────────────────────────────────────
            # beta del muón al llegar al nivel del detector
            theta   = mu['theta']
            gamma_m = E_arr / M_MUON
            beta_m  = np.sqrt(max(1.0 - 1.0/gamma_m**2, 0.0))
            
            # distancia recorrida entre los dos planos
            
            L_trav  = self.geo.sep / np.cos(theta)
            tof_true = L_trav / (beta_m * C_LIGHT)   # ns

            # 4. Deposición de energía  ────────────────────────────
            Edep_t = Deposicion.sample(E_arr, trk['L_top'],    self.geo.top)
            E_mid  = max(E_arr - Edep_t, M_MUON + 0.5)
            Edep_b = Deposicion.sample(E_mid, trk['L_bottom'], self.geo.bot)
            if Edep_t < 0.01 or Edep_b < 0.01:
                continue

            # centellador ─────────────────────────────────────────────
            ph_top = self.sci.t_foton(Edep_t)
            ph_bot = self.sci.t_foton(Edep_b)

            # PMT ──────────────────────────────────────────────────────
            # t0=0 para TOP, t0=tof_true para BOTTOM (desplazamiento FÍSICO)
            t_top, a_top = self.pmt.detectar(ph_top, t0=0.0)
            t_bot, a_bot = self.pmt.detectar(ph_bot, t0=tof_true)
            if len(t_top) == 0 or len(t_bot) == 0:
                cnts['npe'] += 1; continue

            # 7. Waveforms ─────────────────────────────────────────────────
            wf_top = self.elec.forma(t_top, a_top)
            wf_bot = self.elec.forma(t_bot, a_bot)

            # 8. Threshold crossing (extracción de tiempo independiente) ───
            t_t_meas = self.elec.threshold_tiempo(wf_top)
            t_b_meas = self.elec.threshold_tiempo(wf_bot)

            # 9. Trigger ───────────────────────────────────────────────────
            if not self.trig.aceptar(t_t_meas, t_b_meas):
                continue
            cnts['trig'] += 1

            # 10. ToF medido y reconstrucción ─────────────────────────────
            dt_meas   = t_b_meas - t_t_meas
            
            theta_rec = self.reco.angle(dt_meas)

            # 11. Observables de la waveform ──────────────────────────────
            q_top = self.elec.carga(wf_top)
            q_bot = self.elec.carga(wf_bot)
            pk_t  = self.elec.peak(wf_top)
            pk_b  = self.elec.peak(wf_bot)

            # Guardar ──────────────────────────────────────────────────────
            b = self.buf
            b['theta_true'].append(theta)
            b['theta_rec'].append(theta_rec if theta_rec is not None else np.nan)
            b['tof_true'].append(tof_true)
            b['tof_rec'].append(dt_meas)
            b['E_dep_top'].append(Edep_t)
            b['E_dep_bot'].append(Edep_b)
            b['n_pe_top'].append(len(t_top))
            b['n_pe_bot'].append(len(t_bot))
            b['charge_top'].append(q_top)
            b['charge_bot'].append(q_bot)
            b['amp_top'].append(pk_t)
            b['amp_bot'].append(pk_b)

            if len(self.wf_examples) < 4:
                self.wf_examples.append(
                    (wf_top.copy(), wf_bot.copy(), tof_true, dt_meas,
                     Edep_t, len(t_top)))

        # Convertir a numpy 
        for k in self.buf:
            self.buf[k] = np.array(self.buf[k], dtype=float)
        self.buf['t_axis'] = self.elec.t_ax

        n_trig = cnts['trig']
        print()
        print("=" * 62)
        print("  RESUMEN")
        print("=" * 62)
        print(f"  Generados:          {self.n}")
        print(f"  Perdidos en atm.:   {cnts['atm']}  "
              f"({100*cnts['atm']/self.n:.1f}%)")
        print(f"  Pérd. geométrica:   {cnts['geo']}  "
              f"({100*cnts['geo']/self.n:.1f}%)")
        print(f"  Sin PE:             {cnts['npe']}  "
              f"({100*cnts['npe']/self.n:.1f}%)")
        print(f"  Triggerados:        {n_trig}  "
              f"({100*n_trig/self.n:.1f}%)")
        if n_trig > 0:
            b = self.buf
            valid = ~np.isnan(b['theta_rec'])
            dt_res = b['tof_rec'] - b['tof_true']
            print(f"  θ_true media:       {np.degrees(np.mean(b['theta_true'])):.2f}°")
            print(f"  ToF_true media:     {np.mean(b['tof_true']):.2f} ns")
            print(f"  Res. ToF (σ):       {np.std(dt_res):.3f} ns")
            print(f"  E_dep TOP media:    {np.mean(b['E_dep_top']):.2f} MeV")
            print(f"  N_PE TOP media:     {np.mean(b['n_pe_top']):.1f}")
        print()
        return self.buf


# =============================================================================
# MÓDULO 12 — ANÁLISIS Y FIGURAS
# =============================================================================

class Analysis:
  
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['legend.labelcolor'] = 'white'
    plt.rcParams['axes.titlecolor'] = 'black'

    DARK = 'white'; AX = 'white'; TXT = 'black'
    TK   = '#pink'; GR = '#F7FAFF'; TI  = 'black'
    C = [
        '#FF69B4',  # hot pink
        '#FFB5F0',  # rosa pastel
        '#79c0ff',  # celeste
        '#A7D8FF',  # celeste pastel
        '#D4FFF7',  # aqua muy claro
        'green'   # verde pastel
        ]   
#####
    def plot_flujo_altitud(self, ax=None):
   
    
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
    
        alturas_km = np.array([1, 5, 10], dtype=float)
    
        flujo = np.array([
            self.sim.flujo_altura['1km'],
            self.sim.flujo_altura['5km'],
            self.sim.flujo_altura['10km']
        ], dtype=float) / self.sim.n
    
        # --- plot principal ---
        ax.plot(
            alturas_km,
            flujo,
            marker='o',
            lw=2,
            color=self.C[2],
        )
    
        ax.set_xlabel('Altitud (km)')
        ax.set_ylabel('Flujo relativo de muones')
        ax.set_title('')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
        return ax
    



####
    def __init__(self, sim: MuonDetectorSim):
        self.sim   = sim
        self.b     = sim.buf
        valid      = ~np.isnan(self.b['theta_rec'])
        self.tt    = np.degrees(self.b['theta_true'])
        self.tr    = np.degrees(self.b['theta_rec'])
        self.toft  = self.b['tof_true']
        self.tofr  = self.b['tof_rec']

    def _H(self, ax, data, bins, c, label=None, density=False):
        ax.hist(data, bins=bins, color=c, alpha=0.85,
                edgecolor='white', linewidth=0.3,
                density=density, label=label)

    def plot(self, outfile: str) -> str:
      

        if len(self.tt) == 0:
            print("  ERROR: Sin eventos triggerados — no se puede generar figura.")
            return outfile

        fig = plt.figure(figsize=(22, 24))
        gs  = gridspec.GridSpec(4, 4, figure=fig,
                                hspace=0.52, wspace=0.38,
                                top=0.94, bottom=0.04,
                                left=0.06, right=0.96)
        fig.suptitle(
            'Simulación Monte Carlo — Detector de Muones Cósmicos\n'
            'Centelladores BC-408  +  PMT ',
            fontsize=13, fontweight='bold', color=self.TXT)

    
        
        th_r = np.linspace(0, np.pi/2, 300)
        
        
        th_deg = np.degrees(th_r)
        pdf = 3*np.cos(th_r)**2*np.sin(th_r) # flujo angular por elemento de ángulo sólido
        pdf /= np.trapz(pdf, th_deg)
        
        bins_ang = np.linspace(0, 90, 37)

        tof_lo  = max(0.0, min(self.toft.min(), self.tofr.min()) - 0.5)
        tof_hi  = max(self.toft.max(), self.tofr.max()) * 1.05
        bins_tof = np.linspace(tof_lo, tof_hi, 55)

        # ── FILA 0 ──────────────────────────────────────────────────────

        
        ax = fig.add_subplot(gs[0,0])
    
        theta_gen = np.degrees(
        np.array(self.sim.theta_generated)
        )
    
        self._H(ax,
            theta_gen,
            bins_ang,
            self.C[0],
            density=True,
            label='Generados')
        ax.set(xlabel='θ (°)',ylabel='PDF',title='Distribución Angular Generada')

        ax.legend()
        ax.grid(True)
        
        ax.plot(th_deg, pdf, c='black', lw=2, label='cos²θ·sinθ')
    
        ax.legend(fontsize=7); ax.grid(True)

        # 2. θ reconstruida
        ax = fig.add_subplot(gs[0, 1])
        self._H(ax, self.tr, bins_ang,c='pink', density=True, label='Reconstruida')
        #ax.plot(th_deg, pdf, 'w-', lw=2, label='Teórica')
        ax.set(xlabel='θ_rec (°)', ylabel='PDF',
               title='Distribución Angular Reconstruida')
        ax.legend(fontsize=7)
        ax.grid(True)

        # 3. Scatter θ_true vs θ_rec
        ax = fig.add_subplot(gs[0, 2])
        ns = min(len(self.tt), 2500)
        sc = ax.scatter(self.tt[:ns], self.tr[:ns],
                        s=4, alpha=0.45, c=self.toft[:ns], cmap='plasma')
        ax.plot([0,90],[0,90],'b--',lw=1.5, label='Ideal')
        cb = plt.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label('ToF_true (ns)', fontsize=7)
        ax.set(xlabel='θ generados (°)', ylabel='θ reconstruido(°)',
               title='θ generados vs θ recontruidos')
        ax.legend(fontsize=7); ax.grid(True)
        
        ax = fig.add_subplot(gs[0, 3])
        self.plot_flujo_altitud(ax=ax)
       
        # ── FILA 1 ──────────────────────────────────────────────────────

        
        d_c = self.sim.geo.sep / C_LIGHT   # d/c (ns) 

        # 5. ToF verdadero
        ax = fig.add_subplot(gs[1, 0])
        self._H(ax, self.toft, bins_tof, self.C[0])
        ax.axvline(d_c, color='cyan', lw=2, ls='--',
                   label=f'd/c = {d_c:.2f} ns')
        ax.set(xlabel='ToF verdadero (ns)', ylabel='Eventos',
               title='Time of Flight Verdadero')
        ax.legend(fontsize=7); ax.grid(True)

        # 6. ToF medido
        ax = fig.add_subplot(gs[1, 1])
        self._H(ax, self.tofr, bins_tof, self.C[1])
        ax.axvline(d_c, color='cyan', lw=2, ls='--',
                   label=f'd/c = {d_c:.2f} ns')
        ax.set(xlabel='ToF medido (ns)', ylabel='Eventos',
               title='Time of Flight Medido')
        ax.legend(fontsize=7); ax.grid(True)

        # 7. Residuo ToF
        ax = fig.add_subplot(gs[1, 3])
        dtof = self.tofr - self.toft
        mu_t, sg_t = np.mean(dtof), np.std(dtof)
        self._H(ax, dtof, 65, self.C[3])
        ax.axvline(mu_t, color='yellow', lw=2,
                   label=f'μ = {mu_t:.3f} ns')
        ax.axvline(mu_t+sg_t, color='orange', lw=1.5, ls='--',
                   label=f'σ = {sg_t:.3f} ns')
        ax.axvline(mu_t-sg_t, color='orange', lw=1.5, ls='--')
        ax.set(xlabel='ΔToF (ns)', ylabel='Eventos',
               title='Residuo ToF  (ToF_meas − ToF_true)')
        ax.legend(fontsize=7); ax.grid(True)

        # 8. Scatter ToF_true vs ToF_meas
        ax = fig.add_subplot(gs[1, 2])
        ns2 = min(len(self.toft), 2500)
        ax.scatter(self.toft[:ns2], self.tofr[:ns2],
                   s=4, alpha=0.35, color=self.C[5])
        lim = [min(self.toft.min(), self.tofr.min())*0.95,
               max(self.toft.max(), self.tofr.max())*1.05]
        ax.plot(lim, lim, 'b--', lw=1.5, label='Ideal')
        ax.set(xlabel='ToF_true (ns)', ylabel='ToF_meas (ns)',
               title='ToF: Verdadero vs Medido')
        ax.legend(fontsize=7); ax.grid(True)

        # ── FILA 2 ──────────────────────────────────────────────────────

        Edep = self.b['E_dep_top']

        # 9. Energía depositada TOP
        ax = fig.add_subplot(gs[2, 0])
        ep99 = np.percentile(Edep, 99.5)
        eb   = np.linspace(0, ep99, 65)
        self._H(ax, Edep, eb, self.C[3])
        hv, he = np.histogram(Edep, bins=eb)
        mpv_v = 0.5*(he[np.argmax(hv)] + he[np.argmax(hv)+1])
        ax.axvline(mpv_v, color='cyan', lw=2,
                   label=f'MPV = {mpv_v:.2f} MeV')
        ax.axvline(np.mean(Edep), color='yellow', lw=1.5, ls='--',
                   label=f'Media = {np.mean(Edep):.2f} MeV')
        ax.set(xlabel='E_dep TOP (MeV)', ylabel='Eventos',
               title='Energía Depositada (TOP)')
        ax.legend(fontsize=7); ax.grid(True)

        # 10. Número de fotoelectrones
        ax = fig.add_subplot(gs[2, 1])
        npe  = self.b['n_pe_top'].astype(int)
        step = max(1, int(np.mean(npe)/25))
        bpe  = np.arange(0, int(np.percentile(npe,99.5))+step, step)
        self._H(ax, npe, bpe, self.C[0])
        ax.axvline(np.mean(npe), color='yellow', lw=2,
                   label=f'Media = {np.mean(npe):.1f}')
        ax.set(xlabel='N_PE  (capa TOP)', ylabel='Eventos',
               title='Número de Fotoelectrones')
        ax.legend(fontsize=7); ax.grid(True)

        # 11. Amplitud de pico (ambas capas)
        ax = fig.add_subplot(gs[2, 2])
        amps_mV = np.concatenate([self.b['amp_top'],
                                   self.b['amp_bot']]) * 1e3
        self._H(ax, amps_mV, 60, self.C[4])
        ax.set(xlabel='Amplitud pico (mV)', ylabel='Eventos',
               title='Amplitud de Pico  (TOP + BOTTOM)')
        ax.grid(True)

        # 12. Carga integrada (ambas capas)
        ax = fig.add_subplot(gs[2, 3])
        charges = np.concatenate([self.b['charge_top'],
                                   self.b['charge_bot']])
        self._H(ax, charges, 60, self.C[2])
        ax.set(xlabel='Carga integrada (V·ns)', ylabel='Eventos',
               title='Carga Integrada  (TOP + BOTTOM)')
        ax.grid(True)

        # ── FILA 3 — Waveforms ejemplo ──────────────────────────────────
        
        t_ax  = self.b['t_axis']
        mask  = (t_ax >= 0) & (t_ax <= 250)
        wf_c  = [('#58a6ff','#f85149'),
                 ('#3fb950','#e3b341'),
                 ('#bc8cff','#79c0ff'),
                 ('#f0883e','#58a6ff')]

        for ii in range(4):
            ax = fig.add_subplot(gs[3, ii])
            if ii < len(self.sim.wf_examples):
                wt, wb, tof_tr, tof_ms, edep, npe_ev = self.sim.wf_examples[ii]
                ct, cb = wf_c[ii]
                ax.plot(t_ax[mask], wt[mask]*1e3, color=ct,
                        lw=1.2, label='TOP')
                ax.plot(t_ax[mask], wb[mask]*1e3, color=cb,
                        lw=1.2, alpha=0.88, label='BOTTOM')
                ax.axhline(5.0, color='white', lw=0.8, ls='--',
                           alpha=0.7, label='Umbral 5 mV')
                ax.set(xlabel='Tiempo (ns)', ylabel='Amplitud (mV)',
                       title=(f'Waveform — Evento {ii+1}\n'
                              f'ToF_true={tof_tr:.2f} ns | '
                              f'Δt_meas={tof_ms:.2f} ns | '
                              f'E_dep={edep:.2f} MeV | N_PE={npe_ev}'),
                       xlim=(0, 250))
                ax.legend(fontsize=7)
            else:
                ax.text(0.5, 0.5, 'Sin evento disponible',
                        ha='center', va='center',
                        transform=ax.transAxes, color=self.TK)
                ax.set(title=f'Waveform — Evento {ii+1}',
                       xlabel='Tiempo (ns)', ylabel='Amplitud (mV)')
            ax.grid(True)

        plt.savefig(outfile, dpi=150, bbox_inches='tight',
                    facecolor='white')
        plt.close()
        plt.show()
        print(f"  Figura guardada: {outfile}")
        return outfile


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == '__main__':

    sim = MuonDetectorSim(   # cambniar valores aca
        n_muons  = 10000,
        sep_cm   = 30,  
        width_cm = 80,    
        thick_cm = 1,     
    )
    results = sim.run()

    ana     = Analysis(sim) 
    outfile = 'muon_detector_results.png'
    ana.plot(outfile)


    print("  Simulación completada.")
