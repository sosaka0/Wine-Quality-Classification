from pydantic import BaseModel

class wines(BaseModel):
    fixedacidity : float
    volatileacidity : float
    citricacid : float
    residualsugar : float
    chlorides : float
    freesulfurdioxide : float
    totalsulfurdioxide : float
    density : float
    pH :float
    sulphates : float
    alcohol : float

  