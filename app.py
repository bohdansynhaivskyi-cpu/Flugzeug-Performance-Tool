"""
Flugzeug-Performance-Tool
=========================
Interaktive Streamlit-App: Auftrieb, Widerstand, Gleitzahl, Steigrate und
Reichweite (Breguet) auf Basis einer parabolischen Widerstandspolare und der
ISA-Standardatmosphäre.

Start:  streamlit run app.py

ÜBERBLICK: WIE DAS PROGRAMM AUFGEBAUT IST
-----------------------------------------
1. Konstanten          -> Naturkonstanten und ISA-Werte auf Meereshöhe
2. Physik              -> Atmosphäre, Flugzeugklasse mit allen Formeln
3. Vorlagen            -> Beispielflugzeuge (Cessna, ASK 21, A320)
4. Oberflächen-Helfer  -> kleine Funktionen für Diagramme und Kennzahlen
5. Seitenmenü          -> Eingabefelder (Streamlit)
6. Berechnungen        -> ruft die Physik mit den Eingaben auf
7. Anzeige             -> Kennzahlen und Diagramme

WICHTIGE STREAMLIT-REGEL: Bei JEDER Änderung eines Eingabefelds führt Streamlit
das gesamte Skript von oben nach unten neu aus. Deshalb gibt es hier keine
"Berechnen"-Schaltfläche: Alles wird einfach immer neu gerechnet.
"""
# dataclass erzeugt automatisch __init__ & Co. für eine Klasse, die nur Daten hält.
from dataclasses import dataclass

# numpy: schnelle Rechnung mit ganzen Zahlenfeldern (Arrays) statt Schleifen.
import numpy as np
# plotly: interaktive Diagramme (Zoom, Hover mit Zahlenwerten).
import plotly.graph_objects as go
# streamlit: macht aus dem Skript eine Web-Oberfläche.
import streamlit as st

# ----------------------------------------------------------------------------
# Konstanten
# ----------------------------------------------------------------------------
# Alle Rechnungen erfolgen in SI-Einheiten (m, kg, s, N, W). Nur für die
# Anzeige wird in km/h, kt, ft/min usw. umgerechnet. So vermeidet man
# Einheitenfehler, die in der Luftfahrt besonders teuer sind.

G = 9.80665                  # Erdbeschleunigung [m/s²] (Normwert, hier als höhenunabhängig angenommen)
R_AIR = 287.053              # spez. Gaskonstante trockener Luft [J/(kg·K)]; Zustandsgleichung p = rho·R·T
GAMMA = 1.4                  # Isentropenexponent von Luft (cp/cv, zweiatomiges Gas); wird für die Schallgeschwindigkeit gebraucht
T0, P0 = 288.15, 101325.0    # ISA-Meereshöhe: 15 °C = 288,15 K und 1013,25 hPa
RHO0 = P0 / (R_AIR * T0)     # Dichte auf Meereshöhe aus der idealen Gasgleichung p = rho·R·T  ->  ≈ 1,225 kg/m³
MS_TO_FPM = 196.85           # 1 m/s = 196,85 ft/min (Steigrate wird in der Luftfahrt in ft/min angegeben)
MS_TO_KT = 1.94384           # 1 m/s = 1,94384 kt (Knoten = Seemeilen pro Stunde)


# ----------------------------------------------------------------------------
# Physik
# ----------------------------------------------------------------------------
def isa(h):
    """ISA-Standardatmosphäre (0–20 km). Gibt T [K], p [Pa], rho [kg/m³], a [m/s].

    PHYSIK: Die Luft wird immer dünner, je höher man steigt. Dichte, Druck und
    Temperatur bestimmen Auftrieb, Widerstand und Triebwerksleistung. Die
    "International Standard Atmosphere" (ISA) ist ein genormtes Modell mit
    zwei Schichten:

    * Troposphäre (0–11 km): Temperatur sinkt linear um 6,5 K pro km
      (Temperaturgradient -0,0065 K/m).
    * untere Stratosphäre (11–20 km): Temperatur konstant 216,65 K (-56,5 °C).

    PROGRAMMIERUNG: Die Funktion funktioniert mit einer einzelnen Zahl ODER mit
    einem ganzen numpy-Array von Höhen. Dafür dient np.where (siehe unten).
    """
    # Eingabe in ein float-Array verwandeln und auf den Gültigkeitsbereich
    # 0..20000 m begrenzen (clip), damit die Formeln nie außerhalb laufen.
    h = np.clip(np.asarray(h, dtype=float), 0.0, 20000.0)

    # Temperatur: np.where(Bedingung, Wert_wenn_wahr, Wert_wenn_falsch) wählt
    # elementweise. Das ist die Array-Variante eines if/else.
    T = np.where(h <= 11000, T0 - 0.0065 * h, 216.65)

    # Druck:
    # Troposphäre: Aus dem hydrostatischen Gleichgewicht (dp = -rho·g·dh) und
    # dem linearen Temperaturverlauf folgt p = p0·(T/T0)^(g/(R·L)). Der
    # Exponent g/(R·L) mit L = 0,0065 K/m ergibt 5,25588.
    # Stratosphäre: Bei konstanter Temperatur (Isotherme) ergibt dieselbe
    # Gleichung einen exponentiellen Abfall; 22632,06 Pa ist der Druck in
    # 11 km Höhe, an dem die Kurve "angesetzt" wird.
    p = np.where(
        h <= 11000,
        P0 * (T / T0) ** 5.25588,
        22632.06 * np.exp(-G * (h - 11000) / (R_AIR * 216.65)),
    )

    # Rückgabe: Temperatur, Druck, Dichte (ideale Gasgleichung rho = p/(R·T))
    # und Schallgeschwindigkeit a = sqrt(gamma·R·T). Die Schallgeschwindigkeit
    # braucht man für die Machzahl M = V/a.
    return T, p, p / (R_AIR * T), np.sqrt(GAMMA * R_AIR * T)


@dataclass
class Aircraft:
    """Alle Kenndaten eines Flugzeugs plus die zugehörigen Formeln (Methoden)."""
    typ: str                 # "jet" | "prop" | "glider"  (Antriebsart)
    mass: float              # Startmasse [kg]
    fuel: float              # Kraftstoffmasse [kg]
    S: float                 # Flügelfläche [m²] (Bezugsfläche für alle Beiwerte)
    b: float                 # Spannweite [m]
    cd0: float               # Nullwiderstandsbeiwert: Widerstand bei CL = 0 (Reibung, Form, Rumpf, Leitwerk ...)
    e: float                 # Oswald-Faktor: Qualität der Auftriebsverteilung (1 = ideal elliptisch)
    clmax: float             # max. Auftriebsbeiwert: darüber reißt die Strömung ab (Stall)
    thrust_kn: float = 0.0   # Jet: max. statischer Schub (Meereshöhe) [kN]
    power_kw: float = 0.0    # Prop: Wellenleistung (Meereshöhe) [kW]
    eta: float = 0.0         # Prop: Propellerwirkungsgrad
    sfc: float = 0.0         # Jet: g/(kN·s) | Prop: g/kWh  (spezifischer Kraftstoffverbrauch)
    vmax_kmh: float = 300.0  # Obergrenze der Diagramme [km/h TAS]

    @property
    def k(self):
        """Induzierter Widerstandsfaktor k = 1/(π·e·AR) mit AR = b²/S.

        PHYSIK: Ein Flügel, der Auftrieb erzeugt, lenkt Luft nach unten ab.
        An den Flügelspitzen strömt Luft von unten nach oben um (Wirbel). Das
        kostet Energie und wirkt wie ein zusätzlicher Widerstand: der
        INDUZIERTE Widerstand. Er sinkt, wenn der Flügel schlank ist.
        Das Maß dafür ist die Streckung AR (Aspect Ratio) = b²/S.
        Segelflugzeuge haben daher sehr lange, schmale Flügel (AR ≈ 16).

        PROGRAMMIERUNG: @property macht aus der Methode ein "Attribut": man
        schreibt ac.k statt ac.k() und es wird bei Bedarf frisch berechnet.
        """
        return self.S / (np.pi * self.e * self.b ** 2)   # = 1/(π·e·AR), weil AR = b²/S

    def vstall(self, h, m):
        """Überziehgeschwindigkeit Vs [m/s] in Höhe h bei Masse m.

        PHYSIK: Im Horizontalflug gilt Auftrieb = Gewicht:
            m·g = 0,5·rho·V²·S·CL
        Löst man nach V auf und setzt das maximal mögliche CL (= CL,max) ein,
        erhält man die kleinste Geschwindigkeit, bei der noch Horizontalflug
        möglich ist. Langsamer -> Strömungsabriss (Stall).
        Wichtig: Vs steigt mit der Masse und mit der Höhe (dünnere Luft!).
        """
        rho = isa(h)[2]   # Index 2 = Dichte aus dem Tupel (T, p, rho, a)
        return np.sqrt(2 * m * G / (rho * self.S * self.clmax))

    def speeds(self, h, m, n=300, f=1.0):
        """Geschwindigkeitsraster von f·Vs bis zur Diagramm-Obergrenze.

        PROGRAMMIERUNG: Für die Kurven brauchen wir viele Geschwindigkeiten.
        np.linspace(a, b, n) erzeugt n gleichmäßig verteilte Werte von a bis b.
        Untergrenze = f·Vs (f = 1 -> Stallgeschwindigkeit; f = 1,1 -> 10 %
        Sicherheitsabstand). Die Obergrenze ist mindestens 1,3·Untergrenze,
        damit das Intervall nie leer oder verkehrt herum ist (z. B. in großer
        Höhe, wo Vs sehr hoch liegt).
        """
        lo = f * self.vstall(h, m)
        return np.linspace(lo, max(self.vmax_kmh / 3.6, 1.3 * lo), n)

    def level_flight(self, v, h, m):
        """Horizontalflug (L = G): gibt CL, CD, Widerstand D [N].

        PHYSIK Schritt für Schritt:
        1. Staudruck q = 0,5·rho·V². Das ist die "Energiedichte" der
           Anströmung; Auftrieb und Widerstand sind proportional dazu.
        2. Im Horizontalflug muss der Auftrieb das Gewicht tragen:
               L = q·S·CL = m·g   ->   CL = m·g/(q·S)
           Wird man langsamer, sinkt q, also muss CL steigen (größerer
           Anstellwinkel), bis bei CL,max der Stall kommt.
        3. Parabolische Widerstandspolare:
               CD = CD0 + k·CL²
           CD0 = Reibungs-/Formwiderstand (wächst mit V²).
           k·CL² = induzierter Widerstand (wächst bei kleiner V, weil CL
           dann groß sein muss).
        4. Widerstandskraft D = q·S·CD.
        """
        q = 0.5 * isa(h)[2] * v ** 2        # Staudruck [Pa]
        cl = m * G / (q * self.S)           # nötiger Auftriebsbeiwert
        cd = self.cd0 + self.k * cl ** 2    # Widerstandsbeiwert aus der Polare
        return cl, cd, q * self.S * cd      # D in Newton

    def best_glide(self, h, m):
        """Geschwindigkeit und Gleitzahl bei CL = sqrt(CD0/k).

        PHYSIK: Die Gleitzahl E = L/D = CL/CD sagt, wie weit man pro Meter
        Höhenverlust gleiten kann (E = 20 -> 20 m weit pro 1 m Sinken).
        Mit CD = CD0 + k·CL² ist E(CL) = CL/(CD0 + k·CL²). Ableiten und
        gleich null setzen ergibt das Optimum bei CD0 = k·CL², also
            CL,opt = sqrt(CD0/k)
        Dort ist der schädliche Widerstand genau so groß wie der induzierte.
        Eingesetzt ergibt sich E,max = 1/(2·sqrt(CD0·k)) = CL,opt/(2·CD0).
        Die Geschwindigkeit dazu folgt aus L = G mit CL = CL,opt.
        """
        cl = np.sqrt(self.cd0 / self.k)                         # optimaler Auftriebsbeiwert
        v = np.sqrt(2 * m * G / (isa(h)[2] * self.S * cl))      # zugehörige Geschwindigkeit
        return v, cl / (2 * self.cd0)                           # (V, E,max)

    def power_available(self, v, h):
        """Verfügbare Vortriebsleistung [W].

        PHYSIK: Leistung = Kraft · Geschwindigkeit. Für den Vortrieb:
            P_verfügbar = Schub · V
        Strahltriebwerke und Propeller verhalten sich sehr verschieden:
        * Jet: liefert näherungsweise einen SCHUB, der mit der Höhe (Luftdichte)
          fällt. Die Leistung Schub·V steigt daher mit V.
        * Propeller: liefert eine (fast) konstante LEISTUNG, der Schub fällt
          dann mit zunehmender V.
        * Segelflugzeug: kein Antrieb -> 0.
        """
        v = np.asarray(v, dtype=float)   # sicherstellen, dass v ein numpy-Array/Zahl ist
        if self.typ == "jet":
            # a = Schallgeschwindigkeit, p = Druck in Flughöhe (Indizes 1 und 3)
            _, p, _, a = isa(h)
            mach = v / a                 # Machzahl = Fluggeschwindigkeit / Schallgeschwindigkeit
            # Vereinfachtes Schubmodell für Hochnebenstromtriebwerke (A320 usw.):
            #   T = T0 · (p/p0) · (1 + 0,2·M²)^3,5 · (1 - 0,49·√M)
            # * thrust_kn*1e3        : kN in N umrechnen (statischer Schub am Boden)
            # * (p / P0)             : Schub ist ungefähr proportional zum
            #                          Umgebungsdruck (= zur Luftdichte), da ein
            #                          Triebwerk Luftmasse durchsetzt.
            # * (1 + 0,2·M²)^3,5     : Staudruckerholung im Einlauf (isentrope
            #                          Totaldruck-Beziehung); mit höherer Fluggeschwindigkeit
            #                          wird die Luft vorverdichtet, das hebt den Schub.
            # * (1 - 0,49·√M)        : Abnahme des Nettoschubs mit der
            #                          Geschwindigkeit bei großem Nebenstromverhältnis
            #                          (der Impuls der einströmenden Luft wird abgezogen).
            thrust = (self.thrust_kn * 1e3 * (p / P0)
                      * (1 + 0.2 * mach ** 2) ** 3.5 * (1 - 0.49 * np.sqrt(mach)))
            # np.maximum(.., 0): Schub kann nicht negativ werden (Sicherheit bei hoher Mach).
            # Leistung = Schub · V
            return np.maximum(thrust, 0.0) * v
        if self.typ == "prop":
            # Dichteverhältnis sigma = rho/rho0 (1 am Boden, <1 in der Höhe)
            sigma = isa(h)[2] / RHO0
            # Ein Saugmotor verliert mit der Höhe Leistung, weil weniger Luft (Sauerstoff)
            # zur Verbrennung zur Verfügung steht. Faustformel nach Gagg-Farrar:
            #   P/P0 = (sigma - 0,12)/0,88
            # (bei sigma = 1 ergibt das 1, bei sigma = 0,12 null Leistung).
            # Die an den Luftstrom abgegebene Leistung ist Wellenleistung · eta_prop.
            # "+ 0 * v" ist ein Programmier-Trick: Das Ergebnis ist eigentlich ein
            # einzelner Wert (unabhängig von V), soll aber die Form des Arrays v
            # haben, damit Plots und Rechnungen mit gleich langen Feldern funktionieren.
            return self.eta * self.power_kw * 1e3 * np.maximum((sigma - 0.12) / 0.88, 0.0) + 0 * v
        # Segelflugzeug: keine Antriebsleistung (Nullen in gleicher Form wie v)
        return np.zeros_like(v)

    def climb_rate(self, v, h, m):
        """Steigrate [m/s] = Leistungsüberschuss / Gewicht (negativ = Sinken).

        PHYSIK: Die Energiebilanz sagt: Was der Antrieb mehr liefert, als der
        Widerstand im Horizontalflug "verbraucht", kann in Höhengewinn
        (potenzielle Energie m·g·h) umgesetzt werden:
            P_verfügbar - D·V = m·g·(dh/dt)
        also  dh/dt = (P_verf - D·V)/(m·g).
        D·V ist die "benötigte Leistung". Für ein Segelflugzeug ist
        P_verf = 0, die Steigrate ist somit negativ, d. h. es sinkt
        (Sinkrate = Leistung, die der Widerstand "kostet", geteilt durch Gewicht).
        Näherung: kleiner Bahnwinkel (cos γ ≈ 1), L ≈ G. Passt für normale Steigflüge.
        """
        drag = self.level_flight(v, h, m)[2]    # Index 2 = Widerstand D
        return (self.power_available(v, h) - drag * v) / (m * G)

    def breguet_range(self, v, h, m0, m1):
        """Reichweite [m] nach Breguet, L/D bei mittlerer Masse.

        PHYSIK: Je mehr Kraftstoff verbrannt wird, desto leichter wird das
        Flugzeug. Breguet integriert über diese Massenabnahme (von m0 am
        Start auf m1 bei der Landung):
            Jet:       R = V/(cT·g) · (L/D) · ln(m0/m1)
            Propeller: R = eta_p/(cP·g) · (L/D) · ln(m0/m1)
        * cT = Schubspezifischer Verbrauch (kg/(N·s)), cP = leistungsspez. Verbrauch (kg/(J)).
        * Beim Jet geht V ein, weil der Verbrauch an den SCHUB gekoppelt ist:
          schneller fliegen = mehr Strecke pro Sekunde bei gleichem Schubbedarf.
        * Beim Propeller geht V nicht direkt ein (Verbrauch pro LEISTUNG),
          dafür der Propellerwirkungsgrad.
        * ln(m0/m1) = logarithmischer Effekt der abnehmenden Masse.
        Maximale Reichweite also: hohe Gleitzahl, geringer Verbrauch, viel Kraftstoff.

        PROGRAMMIERUNG: L/D wird hier der Einfachheit halber bei der MITTLEREN
        Masse berechnet (0,5·(m0 + m1)) statt exakt integriert.
        """
        cl, cd, _ = self.level_flight(v, h, 0.5 * (m0 + m1))   # CL, CD bei mittlerer Masse
        ln_w = np.log(m0 / m1)                                  # ln(Startmasse / Endmasse)
        if self.typ == "jet":
            # sfc [g/(kN·s)]: * 1e-6 -> kg/(N·s).  (1 g = 1e-3 kg, 1 kN = 1e3 N -> 1e-6)
            return v / (self.sfc * 1e-6 * G) * (cl / cd) * ln_w
        # sfc [g/kWh] -> kg/J:  1 g = 1e-3 kg,  1 kWh = 3,6e6 J  ->  / 3,6e9
        return self.eta / (self.sfc / 3.6e9 * G) * (cl / cd) * ln_w


def ceiling(alts, roc, limit):
    """Höhe, in der die max. Steigrate erstmals unter `limit` fällt.

    PHYSIK / FACHBEGRIFFE: Mit zunehmender Höhe nimmt der Leistungs-
    überschuss ab (dünne Luft: weniger Triebwerksleistung, höherer
    Widerstandsbedarf bei gleicher TAS). Irgendwann reicht er nicht mehr zum
    Steigen.
    * Absolute Gipfelhöhe: Steigrate = 0 m/s (limit = 0)
    * Dienstgipfelhöhe: Steigrate = 100 ft/min = 0,508 m/s. Das ist die
      praktisch nutzbare Höhe, weil man sich der absoluten Gipfelhöhe nur
      unendlich langsam nähert.

    PROGRAMMIERUNG: Gesucht ist der erste Index, ab dem roc <= limit.
    Dazwischen wird linear interpoliert, damit das Ergebnis nicht nur ein
    Höhenwert aus dem 250-m-Raster ist.
    """
    below = np.where(roc <= limit)[0]      # Indizes aller Höhen, in denen die Steigrate zu klein ist
    if len(below) == 0:
        return None                        # Grenze wird im ganzen Bereich nicht unterschritten
    i = below[0]                           # erster Index unter der Grenze
    if i == 0:
        return float(alts[0])              # schon am Boden zu schwach
    # lineare Interpolation zwischen den Punkten (i-1) und (i): Welche Höhe gehört
    # zum Wert `limit`? (np.interp erwartet aufsteigende x-Werte, was hier gilt, da roc[i] < roc[i-1].)
    return float(np.interp(limit, [roc[i], roc[i - 1]], [alts[i], alts[i - 1]]))


# ----------------------------------------------------------------------------
# Flugzeug-Vorlagen (typische Richtwerte, alles im Seitenmenü änderbar)
# ----------------------------------------------------------------------------
# Ein Wörterbuch (dict) von Wörterbüchern: Name -> Kenndaten.
# "vmax" ist die Obergrenze des Geschwindigkeitsreglers/der Diagramme,
# "v" und "h" sind der voreingestellte Betriebspunkt (km/h und m).
PRESETS = {
    "Cessna 172S (Kolbenmotor)": dict(
        typ="prop", mass=1157, fuel=140, S=16.2, b=11.0, cd0=0.034, e=0.70, clmax=1.5,
        thrust_kn=0, power_kw=134, eta=0.70, sfc=275, vmax=320, v=220, h=1500),
    "Schleicher ASK 21 (Segelflugzeug)": dict(
        typ="glider", mass=600, fuel=0, S=17.95, b=17.0, cd0=0.010, e=0.90, clmax=1.5,
        thrust_kn=0, power_kw=0, eta=0, sfc=0, vmax=280, v=100, h=1000),
    "Airbus A320 (Strahltriebwerke)": dict(
        typ="jet", mass=73500, fuel=14000, S=122.6, b=34.1, cd0=0.022, e=0.80, clmax=2.6,
        thrust_kn=240, power_kw=0, eta=0, sfc=16, vmax=950, v=830, h=11000),
}
# Anzeigenamen der Antriebsarten (nur für die Oberfläche)
TYP_NAMES = {"jet": "Strahltriebwerk", "prop": "Propeller", "glider": "kein Antrieb (Segelflug)"}


# ----------------------------------------------------------------------------
# Hilfsfunktionen für die Oberfläche
# ----------------------------------------------------------------------------
def style(fig, title, xlabel, ylabel):
    """Einheitliches Aussehen für alle Diagramme (Titel, Achsen, Höhe, Legende unten)."""
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, height=380,
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25))
    return fig


def marker(fig, x, y, name):
    """Setzt einen rautenförmigen Punkt (z. B. den Betriebspunkt) ins Diagramm."""
    fig.add_scatter(x=[x], y=[y], mode="markers", name=name,
                    marker=dict(size=11, symbol="diamond", line=dict(width=1)))


def show(items):
    """Zeigt eine Reihe von Kennzahlen nebeneinander an.

    items = Liste von (Beschriftung, Wert, Zusatztext).
    st.columns(n) erzeugt n Spalten; zip() verknüpft jede Spalte mit einem Eintrag.
    """
    for col, (label, value, sub) in zip(st.columns(len(items)), items):
        col.metric(label, value)       # große Zahl mit Beschriftung
        if sub:                        # kleiner Zusatztext, nur wenn vorhanden
            col.caption(sub)


# ----------------------------------------------------------------------------
# Seitenmenü: Eingaben
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Flugzeug-Performance-Tool", page_icon="✈️", layout="wide")
st.title("✈️ Flugzeug-Performance-Tool")
st.caption("Auftrieb, Widerstand, Gleitzahl, Steigrate und Reichweite (Breguet). "
           "Werte im Seitenmenü ändern – alle Diagramme aktualisieren sich live.")

sb = st.sidebar                                   # Kurzname für das Seitenmenü
name = sb.selectbox("Flugzeug-Vorlage", list(PRESETS))   # Auswahlliste mit den Vorlagennamen
p = PRESETS[name]                                 # Kenndaten der gewählten Vorlage
typ = p["typ"]                                    # Antriebsart bestimmt, welche Felder angezeigt werden
# Jedes Eingabefeld braucht einen eindeutigen Key. Hängt man den Vorlagennamen an, gilt für jede
# Vorlage ein EIGENES Widget; beim Wechsel der Vorlage werden daher die Standardwerte der neuen
# Vorlage übernommen (und nicht der alte Wert des vorigen Flugzeugs behalten).
wkey = lambda s: f"{s}|{name}"
sb.caption(f"Antriebsart: {TYP_NAMES[typ]}")

sb.subheader("Masse & Geometrie")
# number_input(Beschriftung, Minimum, Maximum, Startwert, Schrittweite, key=...)
mass = sb.number_input("Startmasse (kg)", 100.0, 600000.0, float(p["mass"]), 10.0, key=wkey("mass"))
fuel = 0.0                                        # Segelflugzeug: kein Kraftstoff
if typ != "glider":
    fuel = sb.number_input("Kraftstoffmasse (kg)", 1.0, 300000.0, float(p["fuel"]), 10.0, key=wkey("fuel"))
S = sb.number_input("Flügelfläche S (m²)", 1.0, 1000.0, float(p["S"]), 0.1, key=wkey("S"))
b = sb.number_input("Spannweite b (m)", 1.0, 100.0, float(p["b"]), 0.1, key=wkey("b"))

sb.subheader("Aerodynamik")
cd0 = sb.number_input("Nullwiderstand CD0", 0.005, 0.100, float(p["cd0"]), 0.001, format="%.4f", key=wkey("cd0"))
e = sb.slider("Oswald-Faktor e", 0.50, 1.00, float(p["e"]), 0.01, key=wkey("e"))
clmax = sb.slider("Max. Auftriebsbeiwert CL,max", 1.0, 3.5, float(p["clmax"]), 0.1, key=wkey("clmax"))

# Standardwerte, falls ein Antriebstyp nicht gewählt ist (Variablen müssen später existieren)
thrust = power = eta = sfc = 0.0
if typ == "jet":
    sb.subheader("Antrieb")
    thrust = sb.number_input("Max. Schub gesamt, Meereshöhe (kN)", 1.0, 2000.0, float(p["thrust_kn"]), 5.0, key=wkey("T"))
    sfc = sb.number_input("Spez. Verbrauch TSFC (g/(kN·s))", 5.0, 40.0, float(p["sfc"]), 0.5, key=wkey("sfc"))
elif typ == "prop":
    sb.subheader("Antrieb")
    power = sb.number_input("Wellenleistung (kW)", 1.0, 5000.0, float(p["power_kw"]), 5.0, key=wkey("P"))
    eta = sb.slider("Propellerwirkungsgrad ηp", 0.30, 0.95, float(p["eta"]), 0.01, key=wkey("eta"))
    sfc = sb.number_input("Spez. Verbrauch BSFC (g/kWh)", 150.0, 500.0, float(p["sfc"]), 5.0, key=wkey("sfc"))

sb.subheader("Flugzustand")
h = sb.slider("Flughöhe (m)", 0, 15000, int(p["h"]), 100, key=wkey("h"))
v_kmh = sb.slider("Geschwindigkeit TAS (km/h)", 60, int(p["vmax"]), int(p["v"]), 5, key=wkey("v"))
# TAS = True Air Speed: wahre Fluggeschwindigkeit relativ zur Luft (nicht die Anzeige im Cockpit,
# die in der Höhe wegen geringerer Dichte weniger anzeigt, IAS).

# Plausibilitätsprüfung: Kraftstoff kann nicht schwerer sein als das ganze Flugzeug.
# st.stop() beendet das Skript an dieser Stelle (nichts darunter wird ausgeführt).
if fuel >= mass:
    st.error("Die Kraftstoffmasse muss kleiner als die Startmasse sein.")
    st.stop()

# Aus allen Eingaben das Flugzeugobjekt bauen; die Methoden oben rechnen damit.
ac = Aircraft(typ, mass, fuel, S, b, cd0, e, clmax, thrust, power, eta, sfc, float(p["vmax"]))
powered = typ != "glider"                         # True bei Jet und Propeller

# ----------------------------------------------------------------------------
# Berechnungen
# ----------------------------------------------------------------------------
v = v_kmh / 3.6                                   # km/h -> m/s (durch 3,6 teilen)
W = mass * G                                      # Gewichtskraft [N] = m·g
rho, a = float(isa(h)[2]), float(isa(h)[3])       # Dichte und Schallgeschwindigkeit in Flughöhe
vs = float(ac.vstall(h, mass))                    # Überziehgeschwindigkeit
cl, cd, drag = ac.level_flight(v, h, mass)        # Zustand im gewählten Betriebspunkt
E = cl / cd                                       # Gleitzahl L/D = CL/CD (weil L = q·S·CL, D = q·S·CD)
v_emax, e_max = ac.best_glide(h, mass)            # beste Gleitgeschwindigkeit und beste Gleitzahl
cl_opt = np.sqrt(cd0 / ac.k)                      # CL bei bester Gleitzahl (für das Tangenten-Diagramm)

# Kurven über einem ganzen Geschwindigkeitsbereich bei gewählter Flughöhe
vg = ac.speeds(h, mass)                           # Geschwindigkeitsraster [m/s]
clg, cdg, Dg = ac.level_flight(vg, h, mass)       # CL, CD, D für jede Geschwindigkeit
Pa = ac.power_available(vg, h)                    # verfügbare Leistung für jede Geschwindigkeit
roc_g = ac.climb_rate(vg, h, mass)                # Steigrate für jede Geschwindigkeit (rate of climb)
i_best = int(np.argmax(roc_g))                    # Index des größten Werts (np.argmax)
roc_max, v_roc = roc_g[i_best], vg[i_best]        # beste Steigrate und zugehörige Geschwindigkeit
# Beim Segelflugzeug ist roc_max der "am wenigsten negative" Wert = geringstes Sinken.

m1 = mass - fuel                                  # Landemasse = Startmasse minus verbrannter Kraftstoff (alles verbrannt)
if powered:
    # Steigrate für jede Höhe (0 bis 15 km in 250-m-Schritten). Für jede Höhe wird
    # die maximale Steigrate über alle Geschwindigkeiten gesucht (.max()). Das ist eine
    # "List Comprehension": [Ausdruck for Variable in Liste].
    alts = np.arange(0, 15001, 250)
    roc_alt = np.array([ac.climb_rate(ac.speeds(hh, mass, 150), hh, mass).max() for hh in alts])
    service = ceiling(alts, roc_alt, 0.508)       # Dienstgipfelhöhe: 100 ft/min = 0,508 m/s
    absolute = ceiling(alts, roc_alt, 0.0)        # absolute Gipfelhöhe: Steigrate 0
    range_op = float(ac.breguet_range(v, h, mass, m1))   # Reichweite bei der gewählten Geschwindigkeit

# ----------------------------------------------------------------------------
# Kennwerte
# ----------------------------------------------------------------------------
# Unterhalb von Vs kann der Flügel nicht genug Auftrieb erzeugen: das errechnete CL wäre größer als
# CL,max. Das Modell würde dann trotzdem eine Zahl ausgeben, die aber physikalisch nicht erreichbar
# ist -> deshalb die Warnung.
if v < vs:
    st.warning(f"⚠️ {v_kmh} km/h liegt unter der Überziehgeschwindigkeit ({vs * 3.6:.0f} km/h) – "
               "in diesem Zustand ist kein Horizontalflug möglich.")

st.subheader("Betriebspunkt")
# f"{x:.2f}" formatiert eine Zahl mit 2 Nachkommastellen (f-String).
show([
    ("Auftrieb L", f"{W / 1e3:.2f} kN", "= Gewicht (Horizontalflug)"),   # L = G (Gleichgewicht der Kräfte)
    ("Widerstand D", f"{drag / 1e3:.2f} kN", f"{drag / W * 100:.1f} % des Gewichts"),  # D/W = 1/E
    ("CL", f"{cl:.3f}", None),
    ("CD", f"{cd:.4f}", None),
    ("Gleitzahl L/D", f"{E:.1f}", None),
    ("Machzahl", f"{v / a:.2f}", f"{v * MS_TO_KT:.0f} kt TAS"),
])
# Ab etwa Mach 0,75 steigt der Widerstand durch Verdichtungsstöße stark an ("Wellenwiderstand").
# Das Modell kennt diesen Effekt nicht, daher der Hinweis.
if v / a > 0.75:
    st.caption("Hinweis: Das Modell enthält keinen Wellenwiderstand (Kompressibilität); CD0 gilt als Reiseflugwert.")

st.subheader("Leistungskennwerte")
row = [("Max. Gleitzahl", f"{e_max:.1f}", f"bei {v_emax * 3.6:.0f} km/h"),
       ("Überziehgeschw. Vs", f"{vs * 3.6:.0f} km/h", f"{vs * MS_TO_KT:.0f} kt (bei Startmasse)")]
if powered:
    row += [("Max. Steigrate", f"{roc_max:.1f} m/s", f"{roc_max * MS_TO_FPM:.0f} ft/min bei {v_roc * 3.6:.0f} km/h"),
            ("Dienstgipfelhöhe", f"{service:.0f} m" if service else "> 15000 m",
             f"absolut: {absolute:.0f} m" if absolute else None),
            ("Reichweite (Breguet)", f"{range_op / 1e3:.0f} km", f"{range_op / 1852:.0f} NM bei gewählter Geschw.")]
    # 1 Seemeile (NM) = 1852 m
else:
    row += [("Min. Sinkrate", f"{-roc_max:.2f} m/s", f"bei {v_roc * 3.6:.0f} km/h"),
            # Gleitstrecke = Höhe · Gleitzahl (bei E = 30 sind das 30 m Strecke pro 1 m Höhe)
            ("Gleitstrecke aus h", f"{h * e_max / 1e3:.1f} km", "bei bester Gleitzahl, ohne Wind")]
show(row)

# ----------------------------------------------------------------------------
# Diagramme
# ----------------------------------------------------------------------------
tab_names = ["Polare & Gleitzahl", "Widerstand & Leistung", "Steig-/Sinkrate",
             "Reichweite (Breguet)" if powered else "Gleitstrecke"]
tabs = st.tabs(tab_names)

# --- Tab 1: Polare & Gleitzahl ---------------------------------------------
with tabs[0]:
    c1, c2 = st.columns(2)
    cl_arr = np.linspace(0, clmax, 120)           # CL-Werte von 0 bis CL,max
    fig = go.Figure()
    # Die Polare trägt CL über CD auf. Sie ist eine Parabel: CD = CD0 + k·CL².
    fig.add_scatter(x=cd0 + ac.k * cl_arr ** 2, y=cl_arr, name="Polare CL(CD)")
    # Geometrischer Trick: Die beste Gleitzahl ist die Steigung der Ursprungsgeraden, die die
    # Polare gerade berührt (Tangente). Der Berührpunkt liegt bei CL,opt und CD = 2·CD0.
    # Steigung der Geraden = CL/CD = E,max. f verlängert die Gerade ein Stück über den
    # Berührpunkt hinaus (höchstens bis CL,max, damit sie im Diagramm bleibt).
    f = min(1.4, clmax / cl_opt)
    fig.add_scatter(x=[0, f * 2 * cd0], y=[0, f * cl_opt], mode="lines", line=dict(dash="dash"),
                    name=f"Tangente: L/D max = {e_max:.1f}")
    marker(fig, cd, cl, "Betriebspunkt")
    c1.plotly_chart(style(fig, "Widerstandspolare", "CD", "CL"))

    fig = go.Figure()
    fig.add_scatter(x=vg * 3.6, y=clg / cdg, name="L/D")   # Gleitzahl für jede Geschwindigkeit (m/s -> km/h mit *3,6)
    marker(fig, v_emax * 3.6, e_max, "Bestes Gleiten")
    marker(fig, v_kmh, E, "Betriebspunkt")
    c2.plotly_chart(style(fig, f"Gleitzahl über der Geschwindigkeit (h = {h} m)",
                          "Geschwindigkeit TAS (km/h)", "L/D"))

# --- Tab 2: Widerstand & Leistung ------------------------------------------
with tabs[1]:
    c1, c2 = st.columns(2)
    q = 0.5 * rho * vg ** 2                       # Staudruck für jede Geschwindigkeit
    d_par = q * S * cd0                           # schädlicher Widerstand (parasite drag), wächst ~ V²
    fig = go.Figure()
    fig.add_scatter(x=vg * 3.6, y=d_par / 1e3, name="Schädlicher Widerstand", line=dict(dash="dot"))
    # Der induzierte Widerstand ist Gesamtwiderstand minus schädlicher Widerstand; er sinkt
    # ~ 1/V² mit steigender Geschwindigkeit (weil CL kleiner wird). Die Summe beider hat ein
    # Minimum -> das ist der Punkt des besten Gleitens (D,min bei E,max).
    fig.add_scatter(x=vg * 3.6, y=(Dg - d_par) / 1e3, name="Induzierter Widerstand", line=dict(dash="dot"))
    fig.add_scatter(x=vg * 3.6, y=Dg / 1e3, name="Gesamtwiderstand", line=dict(width=3))
    if powered:
        # Schub = Leistung / V (aus P = T·V)
        fig.add_scatter(x=vg * 3.6, y=Pa / vg / 1e3, name="Verfügbarer Schub", line=dict(width=3))
    c1.plotly_chart(style(fig, f"Widerstand und Schub (h = {h} m)", "Geschwindigkeit TAS (km/h)", "Kraft (kN)"))

    fig = go.Figure()
    # Benötigte Leistung P = D·V. Ihr Minimum liegt bei LANGSAMERER Geschwindigkeit als das
    # Widerstandsminimum: dort gibt es die geringste Sinkrate / die beste Ausdauer.
    fig.add_scatter(x=vg * 3.6, y=Dg * vg / 1e3, name="Benötigte Leistung", line=dict(width=3))
    if powered:
        fig.add_scatter(x=vg * 3.6, y=Pa / 1e3, name="Verfügbare Leistung", line=dict(width=3))
    c2.plotly_chart(style(fig, "Leistungskurven", "Geschwindigkeit TAS (km/h)", "Leistung (kW)"))
    # Steigen ist dort möglich, wo die verfügbare Kurve ÜBER der benötigten liegt; der
    # größte senkrechte Abstand zwischen beiden entspricht der maximalen Steigrate.

# --- Tab 3: Steig-/Sinkrate ------------------------------------------------
with tabs[2]:
    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=vg * 3.6, y=roc_g, name="Steig-/Sinkrate", line=dict(width=3))
    fig.add_hline(y=0, line_dash="dot")           # waagerechte Nulllinie: darüber Steigen, darunter Sinken
    marker(fig, v_roc * 3.6, roc_max, "Maximum" if powered else "Minimales Sinken")
    c1.plotly_chart(style(fig, f"Vertikalgeschwindigkeit (h = {h} m)", "Geschwindigkeit TAS (km/h)", "m/s"))
    if powered:
        fig = go.Figure()
        # Höhe auf der y-Achse, Steigrate auf der x-Achse (so wie ein Piloten-Handbuch es zeigt).
        # Die Kurve läuft mit der Höhe nach links und trifft bei der Gipfelhöhe die Nulllinie.
        fig.add_scatter(y=alts, x=roc_alt, name="Max. Steigrate", line=dict(width=3))
        fig.add_vline(x=0, line_dash="dot")
        if service:
            fig.add_hline(y=service, line_dash="dash", annotation_text=f"Dienstgipfelhöhe {service:.0f} m")
        c2.plotly_chart(style(fig, "Max. Steigrate über der Höhe", "Steigrate (m/s)", "Höhe (m)"))
    else:
        c2.info("Ohne Antrieb ist die Vertikalgeschwindigkeit immer negativ (Sinken). Der höchste Punkt der "
                "Kurve ist das geringste Sinken (für Thermik). Bestes Gleiten liegt bei höherer Geschwindigkeit.")

# --- Tab 4: Reichweite bzw. Gleitstrecke -----------------------------------
with tabs[3]:
    if powered:
        c1, c2 = st.columns(2)
        # Linkes Diagramm: Wie ändert sich die Reichweite, wenn mehr/weniger Kraftstoff getankt
        # wird? Die Landemasse m1 bleibt gleich, die Startmasse wird m1 + Kraftstoff.
        fuels = np.linspace(0, fuel, 60)
        fig = go.Figure()
        fig.add_scatter(x=fuels, y=ac.breguet_range(v, h, m1 + fuels, m1) / 1e3, name="Reichweite", line=dict(width=3))
        marker(fig, fuel, range_op / 1e3, "Gewählte Kraftstoffmasse")
        c1.plotly_chart(style(fig, "Reichweite über der Kraftstoffmasse (Landemasse konstant)", "Kraftstoff (kg)", "Reichweite (km)"))
        # Die Kurve ist leicht gekrümmt wegen des Logarithmus ln(m0/m1): die ersten Kilogramm
        # Kraftstoff bringen mehr Reichweite als die letzten (zusätzlicher Kraftstoff muss
        # selbst mitgeschleppt werden).

        # Rechtes Diagramm: Reichweite über der Geschwindigkeit. f = 1,1: erst ab 110 % der
        # Überziehgeschwindigkeit, damit man nicht im Stallbereich rechnet.
        vr = ac.speeds(h, mass, 200, f=1.1)
        rr = ac.breguet_range(vr, h, mass, m1) / 1e3
        j = int(np.argmax(rr))                    # Index der größten Reichweite
        fig = go.Figure()
        fig.add_scatter(x=vr * 3.6, y=rr, name="Reichweite", line=dict(width=3))
        marker(fig, vr[j] * 3.6, rr[j], "Maximale Reichweite")
        marker(fig, v_kmh, range_op / 1e3, "Betriebspunkt")
        c2.plotly_chart(style(fig, "Reichweite über der Geschwindigkeit", "Geschwindigkeit TAS (km/h)", "Reichweite (km)"))
        # Beim Propeller liegt das Maximum bei (L/D)max (also bei bester Gleitgeschwindigkeit).
        # Beim Jet ist es schneller, weil dort V/(L/D-abhängiger Widerstand) maßgebend ist: das
        # Optimum liegt bei CL = sqrt(CD0/(3k)), also bei ca. 1,32 x der Geschwindigkeit des besten Gleitens.
        st.success(f"Maximale Reichweite: **{rr[j]:.0f} km** bei **{vr[j] * 3.6:.0f} km/h** "
                   f"(gewählt: {range_op / 1e3:.0f} km). "
                   f"Gleitstrecke bei Triebwerksausfall aus {h} m: ca. **{h * e_max / 1e3:.0f} km**.")
    else:
        # Segelflugzeug: Strecke = Höhe · Gleitzahl, also eine Gerade durch den Ursprung.
        hs = np.linspace(0, max(h, 1000) * 1.5, 60)   # Ausgangshöhen von 0 bis 1,5 x aktuelle Höhe
        fig = go.Figure()
        fig.add_scatter(x=hs, y=hs * e_max / 1e3, name=f"Beste Gleitzahl ({e_max:.1f})", line=dict(width=3))
        fig.add_scatter(x=hs, y=hs * E / 1e3, name=f"Gewählte Geschwindigkeit ({E:.1f})", line=dict(width=3))
        marker(fig, h, h * e_max / 1e3, "Aktuelle Höhe")
        st.plotly_chart(style(fig, "Gleitstrecke über der Ausgangshöhe", "Höhe über Grund (m)", "Strecke (km)"))
        st.caption("Annahmen: ruhige Luft, kein Wind, konstante Gleitzahl.")

# Einklappbarer Bereich mit allen Formeln (LaTeX) und den Modellgrenzen.
with st.expander("Formeln & Annahmen"):
    st.latex(r"C_D = C_{D0} + k\,C_L^2,\qquad k=\frac{1}{\pi\,e\,AR},\qquad AR=\frac{b^2}{S}")
    st.latex(r"L=\tfrac12\rho V^2 S\,C_L,\qquad D=\tfrac12\rho V^2 S\,C_D,\qquad V_s=\sqrt{\frac{2mg}{\rho S C_{L,max}}}")
    st.latex(r"\left(\tfrac{L}{D}\right)_{max}=\frac{1}{2\sqrt{C_{D0}\,k}}\quad\text{bei}\quad C_L=\sqrt{C_{D0}/k}")
    st.latex(r"\text{Steigrate}=\frac{P_{verf}-D\,V}{m\,g}")
    st.latex(r"R_{Jet}=\frac{V}{c_T\,g}\,\frac{L}{D}\,\ln\frac{m_0}{m_1},\qquad "
             r"R_{Prop}=\frac{\eta_p}{c_P\,g}\,\frac{L}{D}\,\ln\frac{m_0}{m_1}")
    st.markdown("""
**Annahmen und Grenzen**
- Parabolische Polare mit konstantem CD0, kein Wellenwiderstand, keine Reynolds-Effekte.
- Horizontalflug (L = G); Steigflug in Kleinwinkelnäherung, Steigrate mit Startmasse.
- Jet: Schub nach vereinfachtem Hochnebenstrom-Modell `T = T₀·δ₀·(1 − 0,49·√M)`; Propeller: konstanter Wirkungsgrad, Leistungsabfall nach Gagg-Farrar.
- Breguet-Reichweite mit L/D bei mittlerer Masse, ohne Steig-/Sinkflug, Reserven und Wind – sie ist daher eine Obergrenze.
- Die Ergebnisse sind Näherungen für Lehr- und Vergleichszwecke, **keine** Flugplanung.
""")
