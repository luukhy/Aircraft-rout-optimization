# Matthias Schäfer, Martin Strohmeier, Vincent Lenders, Ivan Martinovic and Matthias Wilhelm.
# "Bringing Up OpenSky: A Large-scale ADS-B Sensor Network for Research".
# In Proceedings of the 13th IEEE/ACM International Symposium on Information Processing in Sensor Networks (IPSN), pages 83-94, April 2014.
import time
from opensky_api import OpenSkyApi
api = OpenSkyApi()

while(True):
    s = api.get_states(time_secs = 0, icao24 = "48ae82")
    print(s.states[0].icao24)
    time.sleep(10)