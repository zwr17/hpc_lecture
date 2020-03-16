#include <iostream>
#include <cmath>

struct Body {
  double w;
  double x;
  double y;
  double z;
};

int main() {
  int n = 20;
  Body *body = new Body [n];
  for (int i=0; i<n; i++) {
    body[i].w = drand48();
    body[i].x = drand48();
    body[i].y = drand48();
    body[i].z = drand48();
  }
  for (int i=0; i<n; i++) {
    double p = 0;
    for (int j=0; j<n; j++) {
      double dx = body[i].x - body[j].x;
      double dy = body[i].y - body[j].y;
      double dz = body[i].z - body[j].z;
      double r2 = dx*dx + dy*dy + dz*dz + 1e-3;
      p += body[j].w / std::sqrt(r2);
    }
    std::cout << i << " " << p << std::endl;
  }
}
