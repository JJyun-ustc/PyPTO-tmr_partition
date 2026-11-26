#include "<RUNTIME_WORKTREE>/results/qkv-round2-20261125/validation/exported-model.h"
#include <cstdio>
int main(){int k[3]={-1,4,-1};double x[12]{},t[3][5]{},y[2];simpler::tmr::learned::residual(x,t,y);std::printf("%.17g %.17g %.17g",simpler::tmr::learned::runtime(k,1,2,0),y[0],y[1]);}
