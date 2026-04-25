import os
import sys

# Aggiunge src e symbolic al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/symbolic")))

from newclid.api import GeometricSolverBuilder
from newclid.jgex.formulation import JGEXFormulation
from newclid.jgex.problem_builder import JGEXProblemBuilder

def test_sanity():
    # Test 1: Midpoint -> Congruence (Il più semplice in assoluto)
    # Nota: Usiamo la sintassi corretta a : ; b : ; c : midpoint c a b ? cong a c b c
    # In JGEX, il primo punto dopo il nome della costruzione è spesso il punto creato.
    test_p = "a : ; b : ; c : midpoint c a b ? cong a c b c"
    print(f"Testing: {test_p}")
    
    try:
        problem_jgex = JGEXFormulation.from_text(test_p)
        builder = JGEXProblemBuilder().with_problem(problem_jgex)
        setup = builder.build()
        solver = GeometricSolverBuilder().build(setup)
        success = solver.run()
        print(f"Result: {'✅ SOLVED' if success else '❌ FAILED'}")
        if not success:
            print(f"Facts deduced: {len(solver.proof_state.facts)}")
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    test_sanity()
