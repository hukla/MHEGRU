#include <iostream>
#include "MHEAddingProblem.h"

int main() {
    SetNumThreads(16);
    MHEAddingProblem app = MHEAddingProblem();
    app.run();
}
