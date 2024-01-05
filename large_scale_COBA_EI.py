import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


class ExponSyn(bp.Projection):
  def __init__(self, pre, post, delay, prob, g_max, tau, out_label=None):
    super().__init__()
    self.proj = bp.dyn.FullProjAlignPostMg(
      pre,
      delay,
      bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), g_max),
      bp.dyn.Expon.desc(post.num, tau=tau),
      bp.dyn.CUBA.desc(),
      post,
      out_label=out_label
    )


class EINet(bp.Network):
  def __init__(self, num_E, num_I, gEE=0.03, gEI=0.03, gIE=0.335, gII=0.335, p=0.2):
    super().__init__()
    self.E = bp.dyn.LifRefLTC(num_E, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                              V_initializer=bp.init.Normal(-55., 2.))
    self.I = bp.dyn.LifRefLTC(num_I, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                              V_initializer=bp.init.Normal(-55., 2.))
    self.E2E = ExponSyn(self.E, self.E, None, prob=p, g_max=gEE, tau=5., out_label='E')
    self.E2I = ExponSyn(self.E, self.I, None, prob=p, g_max=gEI, tau=5., out_label='E')
    self.I2E = ExponSyn(self.I, self.E, None, prob=p, g_max=gIE, tau=10., out_label='I')
    self.I2I = ExponSyn(self.I, self.I, None, prob=p, g_max=gII, tau=10., out_label='I')

  def update(self, E_bg, I_bg):
    self.E2E()
    self.E2I()
    self.I2E()
    self.I2I()
    self.E(E_bg)
    self.I(I_bg)


class AreaConn(bp.Projection):
  def __init__(self, pre: EINet, post: EINet, conn, delay=None, gEE=0.03, tau=5.):
    super().__init__()
    self.E2E = ExponSyn(pre.E, post.E, delay, prob=conn, g_max=gEE, tau=tau, out_label='F')


class VisualSystem(bp.DynSysGroup):
  def __init__(self, ne, ni, conn, delay, area_names,
               gEE=0.03, gEI=0.03, gIE=0.335, gII=0.335, muEE=.0375, p=0.2):
    super().__init__()
    num_area = conn.shape[0]

    # brain areas
    self.areas = bm.node_list()
    for i in range(num_area):
      print(f'Building area {area_names[i]} ...')
      self.areas.append(EINet(ne, ni, gEE=gEE, gEI=gEI, gII=gII, gIE=gIE, p=p))

    # projections
    self.projections = bm.node_list()
    for i in range(num_area):
      for j in range(num_area):
        if conn[j, i] > 0:
          print(f'Building projection from {area_names[i]} to {area_names[j]} ...')
          proj = AreaConn(self.areas[i], self.areas[j], delay=delay[j, i], conn=conn[j, i], gEE=muEE)
          self.projections.append(proj)

  def update(self, bg, v1_bg):
    for proj in self.projections:
      proj()
    self.areas[0](bg + v1_bg, bg)
    for area in self.areas[1:]:
      area(bg, bg)
    outs = {'E-sps': bm.concatenate([area.E.spike for area in self.areas]),
            'I-sps': bm.concatenate([area.I.spike for area in self.areas])}
    return outs


def sps_show(area_names, sps, run_indices, num_exc, title):
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
  fig.add_subplot(gs[0, 0])
  indices, times = bp.measure.raster_plot(sps, run_indices)
  plt.plot(times, indices, '.', markersize=1)
  plt.yticks(np.arange(len(area_names)) * num_exc + num_exc / 2, area_names)
  plt.ylim(0, len(area_names) * num_exc)
  plt.xlim(0., run_indices[-1])
  plt.xlabel('Time [ms]')
  ax = plt.gca()
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')
  plt.title(title)


def try_large_scale_system():
  area_names = ['V1', 'V2', 'V4', 'DP', 'MT', '8m', '5', '8l', 'TEO', '2', 'F1',
                'STPc', '7A', '46d', '10', '9/46v', '9/46d', 'F5', 'TEpd', 'PBr',
                '7m', '7B', 'F2', 'STPi', 'PROm', 'F7', '8B', 'STPr', '24c']

  # fraction of labeled neurons
  flnMatp = loadmat('Joglekar_2018_data/efelenMatpython.mat')
  conn = np.asarray(flnMatp['flnMatpython'].squeeze())  # fln values..Cij is strength from j to i
  # Distance
  speed = 3.5  # axonal conduction velocity
  distMatp = loadmat('Joglekar_2018_data/subgraphWiring29.mat')
  distMat = distMatp['wiring'].squeeze()  # distances between areas values..
  delayMat = np.asarray(distMat / speed)

  num_exc = 3200
  num_inh = 800

  a = np.asarray([0.10108301, 0.60604239, -0.60977116, -0.33540355, 0.0403722])

  model = VisualSystem(
    num_exc, num_inh, area_names=area_names,
    conn=bm.asarray(conn), delay=bm.asarray(delayMat),
    gEE=a[0], gEI=a[1], gIE=a[2], gII=a[3], muEE=a[4], p=0.1,
  )

  v1_inputs = bp.inputs.section_input([0., 8., 0.], [400., 100., 500.])
  bg_inputs = np.ones(v1_inputs.shape) * 11.0
  run_indices = np.arange(v1_inputs.size)
  outs = bm.for_loop(model.step_run, (run_indices, bg_inputs, v1_inputs), progress_bar=True)
  run_indices = run_indices * bm.get_dt()

  sps_show(area_names, outs['E-sps'], run_indices, num_exc, 'E spikes')
  sps_show(area_names, outs['I-sps'], run_indices, num_inh, 'I spikes')

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
  fig.add_subplot(gs[0, 0])
  plt.plot(run_indices, bp.measure.firing_rate(outs['E-sps'], 10., numpy=True), label='E')
  plt.plot(run_indices, bp.measure.firing_rate(outs['I-sps'], 10., numpy=True), label='I')
  plt.legend()

  plt.show()


if __name__ == '__main__':
  try_large_scale_system()
