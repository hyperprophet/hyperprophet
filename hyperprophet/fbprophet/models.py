# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function
from abc import abstractmethod, ABC
from typing import Tuple
from collections import OrderedDict
from enum import Enum
import pickle
import pkg_resources

import os


class IStanBackend(ABC):
    def __init__(self, logger):
        self.model = self.load_model()
        self.logger = logger

    @staticmethod
    @abstractmethod
    def get_type():
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def fit(self, stan_init, stan_data, **kwargs) -> dict:
        pass

    @abstractmethod
    def sampling(self, stan_init, stan_data, samples, **kwargs) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def build_model(target_dir, model_dir):
        pass


class CmdStanPyBackend(IStanBackend):

    @staticmethod
    def get_type():
        return StanBackendEnum.CMDSTANPY.name

    @staticmethod
    def build_model(target_dir, model_dir):
        from shutil import copy
        import cmdstanpy
        model_name = 'prophet.stan'
        target_name = 'prophet_model.bin'

        sm = cmdstanpy.Model(stan_file=os.path.join(model_dir, model_name))
        sm.compile()
        copy(sm.exe_file, os.path.join(target_dir, target_name))

    def load_model(self):
        import cmdstanpy
        model_file = pkg_resources.resource_filename(
            'fbprophet',
            'stan_model/prophet_model.bin',
        )
        return cmdstanpy.Model(exe_file=model_file)

    def fit(self, stan_init, stan_data, **kwargs):
        (stan_init, stan_data) = self.prepare_data(stan_init, stan_data)
        if 'algorithm' not in kwargs:
            kwargs['algorithm'] = 'Newton' if stan_data['T'] < 100 else 'LBFGS'
        iterations = int(1e4)
        try:
            stan_fit = self.model.optimize(data=stan_data,
                                           inits=stan_init,
                                           iter=iterations,
                                           **kwargs)
        except RuntimeError as e:
            # Fall back on Newton
            if kwargs['algorithm'] != 'Newton':
                self.logger.warning(
                    'Optimization terminated abnormally. Falling back to Newton.'
                )
                kwargs['algorithm'] = 'Newton'
                stan_fit = self.model.optimize(data=stan_data,
                                               inits=stan_init,
                                               iter=iterations,
                                               **kwargs)
            else:
                raise e

        params = self.stan_to_dict_numpy(stan_fit.column_names, stan_fit.optimized_params_np)
        for par in params:
            params[par] = params[par].reshape((1, -1))
        return params

    def sampling(self, stan_init, stan_data, samples, **kwargs) -> dict:
        (stan_init, stan_data) = self.prepare_data(stan_init, stan_data)

        if 'chains' not in kwargs:
            kwargs['chains'] = 4
        if 'warmup_iters' not in kwargs:
            kwargs['warmup_iters'] = samples // 2

        stan_fit = self.model.sample(data=stan_data,
                                     inits=stan_init,
                                     sampling_iters=samples,
                                     **kwargs)
        res = stan_fit.sample
        (samples, c, columns) = res.shape
        res = res.reshape((samples * c, columns))
        params = self.stan_to_dict_numpy(stan_fit.column_names, res)

        for par in params:
            s = params[par].shape
            if s[1] == 1:
                params[par] = params[par].reshape((s[0],))

            if par in ['delta', 'beta'] and len(s) < 2:
                params[par] = params[par].reshape((-1, 1))

        return params

    @staticmethod
    def prepare_data(init, data) -> Tuple[dict, dict]:
        cmdstanpy_data = {
            'T': data['T'],
            'S': data['S'],
            'K': data['K'],
            'tau': data['tau'],
            'trend_indicator': data['trend_indicator'],
            'y': data['y'].tolist(),
            't': data['t'].tolist(),
            'cap': data['cap'].tolist(),
            't_change': data['t_change'].tolist(),
            's_a': data['s_a'].tolist(),
            's_m': data['s_m'].tolist(),
            'X': data['X'].to_numpy().tolist(),
            'sigmas': data['sigmas']
        }

        cmdstanpy_init = {
            'k': init['k'],
            'm': init['m'],
            'delta': init['delta'].tolist(),
            'beta': init['beta'].tolist(),
            'sigma_obs': 1
        }
        return (cmdstanpy_init, cmdstanpy_data)

    @staticmethod
    def stan_to_dict_numpy(column_names: Tuple[str, ...], data: 'np.array'):
        import numpy as np

        output = OrderedDict()

        prev = None

        start = 0
        end = 0
        two_dims = True if len(data.shape) > 1 else False
        for cname in column_names:
            parsed = cname.split(".")

            curr = parsed[0]
            if prev is None:
                prev = curr

            if curr != prev:
                if prev in output:
                    raise RuntimeError(
                        "Found repeated column name"
                    )
                if two_dims:
                    output[prev] = np.array(data[:, start:end])
                else:
                    output[prev] = np.array(data[start:end])
                prev = curr
                start = end
                end += 1
            else:
                end += 1

        if prev in output:
            raise RuntimeError(
                "Found repeated column name"
            )
        if two_dims:
            output[prev] = np.array(data[:, start:end])
        else:
            output[prev] = np.array(data[start:end])
        return output


class PyStanBackend(IStanBackend):

    @staticmethod
    def get_type():
        return StanBackendEnum.PYSTAN.name

    @staticmethod
    def build_model(target_dir, model_dir):
        import pystan
        model_name = 'prophet.stan'
        target_name = 'prophet_model.pkl'
        with open(os.path.join(model_dir, model_name)) as f:
            model_code = f.read()
        sm = pystan.StanModel(model_code=model_code)
        with open(os.path.join(target_dir, target_name), 'wb') as f:
            pickle.dump(sm, f, protocol=pickle.HIGHEST_PROTOCOL)

    def sampling(self, stan_init, stan_data, samples, **kwargs) -> dict:

        args = dict(
            data=stan_data,
            init=lambda: stan_init,
            iter=samples,
        )
        args.update(kwargs)
        stan_fit = self.model.sampling(**args)
        out = dict()
        for par in stan_fit.model_pars:
            out[par] = stan_fit[par]
            # Shape vector parameters
            if par in ['delta', 'beta'] and len(out[par].shape) < 2:
                out[par] = out[par].reshape((-1, 1))
        return out

    def fit(self, stan_init, stan_data, **kwargs) -> dict:

        args = dict(
            data=stan_data,
            init=lambda: stan_init,
            algorithm='Newton' if stan_data['T'] < 100 else 'LBFGS',
            iter=1e4,
        )
        args.update(kwargs)
        try:
            params = self.model.optimizing(**args)
        except RuntimeError:
            # Fall back on Newton
            self.logger.warning(
                'Optimization terminated abnormally. Falling back to Newton.'
            )
            args['algorithm'] = 'Newton'
            params = self.model.optimizing(**args)

        for par in params:
            params[par] = params[par].reshape((1, -1))

        return params

    def load_model(self):
        """Load compiled Stan model"""
        model_file = pkg_resources.resource_filename(
            'fbprophet',
            'stan_model/prophet_model.pkl',
        )
        with open(model_file, 'rb') as f:
            return pickle.load(f)


class StanBackendEnum(Enum):
    PYSTAN = PyStanBackend
    CMDSTANPY = CmdStanPyBackend

    @staticmethod
    def get_backend_class(name: str) -> IStanBackend:
        try:
            return StanBackendEnum[name].value
        except KeyError:
            raise ValueError("Unknown stan backend: {}".format(name))
