#include "pair_bvv.h"
#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"

using namespace LAMMPS_NS;

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

PairBVV::PairBVV(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;

  nelements = 0;
  elements = NULL;
  nparams = maxparam = 0;
  params = NULL;
  elem2param = NULL;
  map = NULL;

  maxshort = 10;
  neighshort = NULL;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairBVV::~PairBVV()
{
  if (copymode) return;

  if (elements)
    for (int i = 0; i < nelements; i++) delete [] elements[i];
  delete [] elements;
  memory->destroy(params);
  memory->destroy(elem2param);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(neighshort);
    delete [] map;
  }
}

/* ---------------------------------------------------------------------- */

void PairBVV::compute(int eflag, int vflag)
{
  int i,j,k,ii,jj,kk,inum,jnum;
  int itype,jtype,ktype,ijparam,ikparam,ijkparam,iiparam;
  tagint itag,jtag;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair, evdwltmp;
  double rsq,rsq1,rsq2,r1,r2,r1inv,r2inv;
  double delr1[3],delr2[3],fj[3],fk[3];
  double force_prefactor;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  double fxtmp,fytmp,fztmp;
  double fjxtmp,fjytmp,fjztmp;
  double fkxtmp,fkytmp,fkztmp;
  double Vi,Vitmp,Wisq;
  double cosine_theta,exp_prefact;

  evdwl = 0.0;
  ev_init(eflag,vflag);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // Beginning of 2-body term
  // loop over full neighbor list of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itag = tag[i];
    itype = map[type[i]];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    fxtmp = fytmp = fztmp = 0.0;
    evdwltmp = 0.0;

    // Force on each ith atom computed separately. No newton imposed

    jlist = firstneigh[i];
    jnum = numneigh[i];
    int numshort = 0;

    iiparam = elem2param[itype][itype][itype];

    // Loop just for the evaluation of Vi
    Vi = 0.0;
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];

      rsq = delx*delx + dely*dely + delz*delz;
      jtype = map[type[j]];
      ijparam = elem2param[itype][jtype][jtype];

      if (rsq >= params[ijparam].cutsq) {
        continue;
      } else {
        neighshort[numshort++] = j;
        if (numshort >= maxshort) {
          maxshort += maxshort/2;
          memory->grow(neighshort,maxshort,"pair:neighshort");
        }
      }

      twobody(&params[ijparam],rsq,fpair,&Vitmp,eflag,evdwltmp);
      Vi += Vitmp;
    }
    // By here, Vi is already evaluated
    // Compute the energy of the system and update the EVtally (only energies, no virial)
    evdwl = pow(Vi - params[iiparam].V0, 2) * params[iiparam].S;
    if (evflag) ev_tally_full(i,evdwl,0.0,0.0,0.0,0.0,0.0);

    // This is a loop to compute only the 2 body forces
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      jtype = map[type[j]];
      ijparam = elem2param[itype][jtype][jtype];
      if (rsq >= params[ijparam].cutsq) {
        continue;
      } else {
        neighshort[numshort++] = j;
        if (numshort >= maxshort) {
          maxshort += maxshort/2;
          memory->grow(neighshort,maxshort,"pair:neighshort");
        }
      }

      twobody(&params[ijparam],rsq,fpair,&Vitmp,eflag,evdwltmp);

      force_prefactor = (Vi - params[iiparam].V0) * (-2*params[iiparam].S);
      fxtmp += delx*fpair*force_prefactor;
      fytmp += dely*fpair*force_prefactor;
      fztmp += delz*fpair*force_prefactor;

      f[i][0] += fxtmp*force_prefactor;
      f[i][1] += fytmp*force_prefactor;
      f[i][2] += fztmp*force_prefactor;
      f[j][0] -= fxtmp*force_prefactor;
      f[j][1] -= fytmp*force_prefactor;
      f[j][2] -= fztmp*force_prefactor;

      if (evflag) ev_tally(i,j,nlocal,newton_pair,0.0,0.0,fpair*force_prefactor,delx,dely,delz);
    }


    // Computing only the bond valence vector sqared, Wisq
    Wisq = 0.0;
    for (jj = 0; jj < numshort; jj++) {
      j = neighshort[jj];
      jtype = map[type[j]];
      ijparam = elem2param[itype][jtype][jtype];
      delr1[0] = x[j][0] - xtmp;
      delr1[1] = x[j][1] - ytmp;
      delr1[2] = x[j][2] - ztmp;
      rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];
      r1 = sqrt(rsq1);

      for (kk = 0; kk < numshort; kk++) {
        k = neighshort[kk];
        ktype = map[type[k]];
        ikparam = elem2param[itype][ktype][ktype];
        ijkparam = elem2param[itype][jtype][ktype];

        delr2[0] = x[k][0] - xtmp;
        delr2[1] = x[k][1] - ytmp;
        delr2[2] = x[k][2] - ztmp;
        rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];
        r2 = sqrt(rsq2);

        Wisq += (pow(params[ijparam].r0,params[ijparam].C0) * pow(params[ikparam].r0,params[ikparam].C0) /
               (pow(r1,params[ijparam].C0) * pow(r2,params[ikparam].C0) ) ) *
        (((delr1[0]*delr2[0])+(delr1[1]*delr2[1])+(delr1[2]*delr2[2]))/(r1*r2));

      } // k-loop ends
    } // j-loop ends

    // By this point, Wisq is computed

    evdwl = params[iiparam].D * ( pow( Wisq - pow(params[iiparam].W0,2), 2) );

    if (evflag) ev_tally_full(i,evdwl,0.0,0.0,0.0,0.0,0.0);


    force_prefactor = (Wisq - pow(params[iiparam].W0,2)) * (-2.0 * params[iiparam].D);

    // Loop to compute 3 body forces only
    for (jj = 0; jj < numshort; jj++) {
      j = neighshort[jj];
      jtype = map[type[j]];
      ijparam = elem2param[itype][jtype][jtype];
      delr1[0] = x[j][0] - xtmp;
      delr1[1] = x[j][1] - ytmp;
      delr1[2] = x[j][2] - ztmp;
      rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];
      r1 = sqrt(rsq1);
      r1inv = 1.0/r1;

      fjxtmp = fjytmp = fjztmp = 0.0;

      for (kk = 0; kk < numshort; kk++) {
        k = neighshort[kk];
        ktype = map[type[k]];
        ikparam = elem2param[itype][ktype][ktype];
        ijkparam = elem2param[itype][jtype][ktype];

        delr2[0] = x[k][0] - xtmp;
        delr2[1] = x[k][1] - ytmp;
        delr2[2] = x[k][2] - ztmp;
        rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];
        r2 = sqrt(rsq2);
        r2inv = 1.0/r2;

        cosine_theta = ((delr1[0]*delr2[0]) + (delr1[1]*delr2[1]) + (delr1[2]*delr2[2])) * r1inv * r2inv;
        exp_prefact = (pow(params[ijparam].r0,params[ijparam].C0) * pow(params[ikparam].r0,params[ikparam].C0) /
                       (pow(r1,params[ijparam].C0) * pow(r2,params[ikparam].C0) ) );

        fj[0] = exp_prefact * cosine_theta * delr1[0] * r1inv * r1inv;
        fj[0] += exp_prefact * delr1[0] * r1inv * r2inv;
        fj[0] -= exp_prefact * cosine_theta * delr1[0] * r1inv * r1inv;
        fjxtmp += fj[0] * force_prefactor;

        fk[0]  = exp_prefact * cosine_theta * delr2[0] * r2inv * r2inv;
        fk[0] += exp_prefact * delr2[0] * r1inv * r2inv;
        fk[0] -= exp_prefact * cosine_theta * delr2[0] * r2inv * r2inv;
        fkxtmp = fk[0] * force_prefactor;

        fj[1] = exp_prefact * cosine_theta * delr1[1] * r1inv * r1inv;
        fj[1] += exp_prefact * delr1[1] * r1inv * r2inv;
        fj[1] -= exp_prefact * cosine_theta * delr1[1] * r1inv * r1inv;
        fjytmp += fj[1] * force_prefactor;

        fk[1]  = exp_prefact * cosine_theta * delr2[1] * r2inv * r2inv;
        fk[1] += exp_prefact * delr2[1] * r1inv * r2inv;
        fk[1] -= exp_prefact * cosine_theta * delr2[1] * r2inv * r2inv;
        fkytmp = fk[1] * force_prefactor;

        fj[2] = exp_prefact * cosine_theta * delr1[2] * r1inv * r1inv;
        fj[2] += exp_prefact * delr1[2] * r1inv * r2inv;
        fj[2] -= exp_prefact * cosine_theta * delr1[2] * r1inv * r1inv;
        fjztmp += fj[2] * force_prefactor;

        fk[2]  = exp_prefact * cosine_theta * delr2[2] * r2inv * r2inv;
        fk[2] += exp_prefact * delr2[2] * r1inv * r2inv;
        fk[2] -= exp_prefact * cosine_theta * delr2[2] * r2inv * r2inv;
        fkztmp = fk[2] * force_prefactor;

        fxtmp -=  fjxtmp + fkxtmp;
        fytmp -=  fjytmp + fkytmp;
        fztmp -=  fjztmp + fkztmp;

        f[k][0] += fkxtmp;
        f[k][1] += fkytmp;
        f[k][2] += fkztmp;

        if (evflag) ev_tally3(i,j,k,0.0,0.0,fj,fk,delr1,delr2);

      }
      f[j][0] += fjxtmp;
      f[j][1] += fjytmp;
      f[j][2] += fjztmp;
    }

    f[i][0] += fxtmp;
    f[i][1] += fytmp;
    f[i][2] += fztmp;
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairBVV::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(neighshort,maxshort,"pair:neighshort");
  map = new int[n+1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairBVV::settings(int narg, char **/*arg*/)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairBVV::coeff(int narg, char **arg)
{
  int i,j,n;

  if (!allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // insure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL
  // nelements = # of unique elements
  // elements = list of element names

  if (elements) {
    for (i = 0; i < nelements; i++) delete [] elements[i];
    delete [] elements;
  }
  elements = new char*[atom->ntypes];
  for (i = 0; i < atom->ntypes; i++) elements[i] = NULL;

  nelements = 0;
  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-2] = -1;
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i],elements[j]) == 0) break;
    map[i-2] = j;
    if (j == nelements) {
      n = strlen(arg[i]) + 1;
      elements[j] = new char[n];
      strcpy(elements[j],arg[i]);
      nelements++;
    }
  }

  // read potential file and initialize potential parameters

  read_file(arg[2]);
  setup_params();

  // clear setflag since coeff() called once with I,J = * *

  n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements

  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairBVV::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style Bond Valence Vector requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style Bond Valence Vector requires newton pair on");

  // need a full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairBVV::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairBVV::read_file(char *file)
{
  int params_per_line = 10;
  char **words = new char*[params_per_line+1];

  memory->sfree(params);
  params = NULL;
  nparams = maxparam = 0;

  // open file on proc 0

  FILE *fp;
  if (comm->me == 0) {
    fp = force->open_potential(file);
    if (fp == NULL) {
      char str[128];
      snprintf(str,128,"Cannot open Bond Valence Vector potential file %s",file);
      error->one(FLERR,str);
    }
  }

  // read each set of params from potential file
  // one set of params can span multiple lines
  // store params if all 3 element tags are in element list

  int n,nwords,ielement,jelement,kelement;
  char line[MAXLINE],*ptr;
  int eof = 0;

  while (1) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fp);
      if (ptr == NULL) {
        eof = 1;
        fclose(fp);
      } else n = strlen(line) + 1;
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    // strip comment, skip line if blank

    if ((ptr = strchr(line,'#'))) *ptr = '\0';
    nwords = atom->count_words(line);
    if (nwords == 0) continue;

    // concatenate additional lines until have params_per_line words

    while (nwords < params_per_line) {
      n = strlen(line);
      if (comm->me == 0) {
        ptr = fgets(&line[n],MAXLINE-n,fp);
        if (ptr == NULL) {
          eof = 1;
          fclose(fp);
        } else n = strlen(line) + 1;
      }
      MPI_Bcast(&eof,1,MPI_INT,0,world);
      if (eof) break;
      MPI_Bcast(&n,1,MPI_INT,0,world);
      MPI_Bcast(line,n,MPI_CHAR,0,world);
      if ((ptr = strchr(line,'#'))) *ptr = '\0';
      nwords = atom->count_words(line);
    }

    if (nwords != params_per_line)
      error->all(FLERR,"Incorrect format in Bond Valence Vector potential file");

    // words = ptrs to all words in line

    nwords = 0;
    words[nwords++] = strtok(line," \t\n\r\f");
    while ((words[nwords++] = strtok(NULL," \t\n\r\f"))) continue;

    // ielement,jelement,kelement = 1st args
    // if all 3 args are in element list, then parse this line
    // else skip to next entry in file

    for (ielement = 0; ielement < nelements; ielement++)
      if (strcmp(words[0],elements[ielement]) == 0) break;
    if (ielement == nelements) continue;
    for (jelement = 0; jelement < nelements; jelement++)
      if (strcmp(words[1],elements[jelement]) == 0) break;
    if (jelement == nelements) continue;
    for (kelement = 0; kelement < nelements; kelement++)
      if (strcmp(words[2],elements[kelement]) == 0) break;
    if (kelement == nelements) continue;

    // load up parameter settings and error check their values

    if (nparams == maxparam) {
      maxparam += DELTA;
      params = (Param *) memory->srealloc(params,maxparam*sizeof(Param),
                                          "pair:params");
    }

    params[nparams].ielement = ielement;
    params[nparams].jelement = jelement;
    params[nparams].kelement = kelement;
    params[nparams].V0 = atof(words[3]);
    params[nparams].W0 = atof(words[4]);
    params[nparams].S = atof(words[5]);
    params[nparams].D = atof(words[6]);
    params[nparams].r0 = atof(words[7]);
    params[nparams].C0 = atof(words[8]);
    params[nparams].rcut = atof(words[9]);

    nparams++;
  }

  delete [] words;
}

/* ---------------------------------------------------------------------- */

void PairBVV::setup_params()
{
  int i,j,k,m,n;
  double rtmp;

  // set elem2param for all triplet combinations
  // must be a single exact match to lines read from file
  // do not allow for ACB in place of ABC

  memory->destroy(elem2param);
  memory->create(elem2param,nelements,nelements,nelements,"pair:elem2param");

  for (i = 0; i < nelements; i++)
    for (j = 0; j < nelements; j++)
      for (k = 0; k < nelements; k++) {
        n = -1;
        for (m = 0; m < nparams; m++) {
          if (i == params[m].ielement && j == params[m].jelement &&
              k == params[m].kelement) {
            if (n >= 0) error->all(FLERR,"Potential file has duplicate entry");
            n = m;
          }
        }
        if (n < 0) error->all(FLERR,"Potential file is missing an entry");
        elem2param[i][j][k] = n;
      }


  // compute parameter values derived from inputs

  // set cutsq using shortcut to reduce neighbor list for accelerated
  // calculations. cut must remain unchanged as it is a potential parameter
  // (cut = a*sigma)

  for (m = 0; m < nparams; m++) {
    params[m].cut = params[m].rcut;
    params[m].cutsq = params[m].cut * params[m].cut;
  }

  // set cutmax to max of all params

  cutmax = 0.0;
  for (m = 0; m < nparams; m++) {
    rtmp = sqrt(params[m].cutsq);
    if (rtmp > cutmax) cutmax = rtmp;
  }
}

/* ---------------------------------------------------------------------- */

void PairBVV::twobody(Param *param, double rsq, double &fforce, double *Vi,
                     int eflag, double &eng)
{
  double r,rinvsq,rp,rq,rainv,rainvsq,expsrainv;

  r = sqrt(rsq);
  rinvsq = 1.0/rsq;
  *Vi = pow(param->r0,param->C0) / pow(r,param->C0);
  fforce = param->C0 * pow(param->r0, param->C0) / pow(r, (param->C0)+2);
  if (eflag) eng = *Vi;
}

/* ---------------------------------------------------------------------- */

void PairBVV::threebody(Param *paramij, Param *paramik, Param *paramijk,
                       double rsq1, double rsq2,
                       double *delr1, double *delr2,
                       double *fj, double *fk, int eflag, double &eng)
{
  return;
}
