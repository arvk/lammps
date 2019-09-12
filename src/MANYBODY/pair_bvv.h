#ifdef PAIR_CLASS

PairStyle(bvv,PairBVV)

#else

#ifndef LMP_PAIR_BVV_H
#define LMP_PAIR_BVV_H

#include "pair.h"

namespace LAMMPS_NS {

class PairBVV : public Pair {
 public:
  PairBVV(class LAMMPS *);
  virtual ~PairBVV();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  virtual double init_one(int, int);
  virtual void init_style();

  struct Param {
    double V0, W0, S, D, r0, C0, rcut;
    double cut, cutsq;
    double costheta;
    int ielement,jelement,kelement;
  };

 protected:
  double cutmax;                // max cutoff for all elements
  int nelements;                // # of unique elements
  char **elements;              // names of unique elements
  int ***elem2param;            // mapping from element triplets to parameters
  int *map;                     // mapping from atom types to elements
  int nparams;                  // # of stored parameter sets
  int maxparam;                 // max # of parameter sets
  Param *params;                // parameter set for an I-J-K interaction
  int maxshort;                 // size of short neighbor list array
  int *neighshort;              // short neighbor list array

  virtual void allocate();
  void read_file(char *);
  virtual void setup_params();
  void twobody(Param *, double, double &, int, double &, double &);
  void threebody(Param *, Param *, Param *, double, double, double *, double *,
                 double *, double *, int, double &);
};

}

#endif
#endif
