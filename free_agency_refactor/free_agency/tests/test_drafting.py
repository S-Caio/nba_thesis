import numpy as np
import pytest

from free_agency.constants import LeagueConfig, RATING, TEAM, AGE, CONTRACT_LEN, SALARY
from free_agency.state import LeagueState
from free_agency.contracts import handle_signing, contract_update