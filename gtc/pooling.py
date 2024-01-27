from collections import defaultdict
from typing import Any, List, Tuple
import einops
import numpy as np
import torch
from escnn.gspaces import *
from escnn.nn import FieldType, GeometricTensor
from escnn.nn.modules.equivariant_module import EquivariantModule
from escnn.nn.modules.invariantmaps import GroupPooling
from escnn.nn.modules.utils import indexes_from_labels
from torch import nn

from gtc.functional import get_cayley_table, buildFplus, first_last_cb, clebsch_gordan, build_Fplus_vectorized


class TCGroupPoolingEfficient(GroupPooling):
    def __init__(self, in_type, group, idx=None, **kwargs):
        super().__init__(in_type, **kwargs)
        self.idx = idx
        self.group = group()
        self.cayley_table = get_cayley_table(self.group)

    def triple_correlation(self, x):
        b, k, d = x.shape
        x = x.reshape((b * k, d))
        nexts = x[:, self.cayley_table]
        mult = x.unsqueeze(1) * x[:, self.cayley_table.swapaxes(0, 1)]
        TC = torch.bmm(mult, nexts)
        TC = TC.reshape((b, k, d, d))
        return TC

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Apply Group Pooling to the input feature map.

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """
        assert input.type == self.in_type

        coords = input.coords
        input = input.tensor
        b, c = input.shape[:2]
        spatial_shape = input.shape[2:]

        for s, contiguous in self._contiguous.items():
            in_indices = getattr(self, "in_indices_{}".format(s))
            out_indices = getattr(self, "out_indices_{}".format(s))

            if contiguous:
                fm = input[:, in_indices[0] : in_indices[1], ...]
            else:
                fm = input[:, in_indices, ...]

            # split the channel dimension in 2 dimensions, separating fields
            fm = fm.view(b, -1, s, *spatial_shape)

            output = self.triple_correlation(fm.squeeze())

            if self.idx is None:
                idx = torch.triu_indices(output.shape[2], output.shape[3])
            else:
                idx = self.idx

            output = output[:, :, idx[0], idx[1]]
            a, b, c = output.shape
            output = output.reshape((a * b, c))
            output = output / (output.norm(dim=0, keepdim=True) + 1e-5)
            output = output.reshape((a, b, c, 1, 1))
        return output

    def export(self):
        raise NotImplementedError


class TCGroupPooling(GroupPooling):
    def __init__(self, in_type, group_type="cyclic", idx=None, **kwargs):
        """
        group_type should be "cyclic" or "dihedral"
        """
        super().__init__(in_type, **kwargs)
        self.idx = idx
        self.group_type = group_type

    def triple_correlation_vectorized_batch_cyclic(self, x):
        b, k, d = x.shape
        x = x.reshape((b * k, d))
        all_rolls = torch.zeros((b * k, d, d)).to(x.device)
        for i in range(d):
            all_rolls[:, :, i] = torch.roll(x, -i, dims=-1)
        rolls_mult = x.unsqueeze(1) * all_rolls
        TC = torch.bmm(rolls_mult, all_rolls)
        TC = TC.reshape((b, k, d, d))
        return TC

    def triple_correlation_vectorized_batch_dihedral(self, x):
        b, k, d = x.shape
        n = d // 2
        x = x.reshape((b * k, d))
        all_rolls = torch.zeros((b * k, d, d)).to(x.device)
        for i in range(d):
            roll0 = torch.roll(x[:, :n], -i, dims=-1)
            roll1 = torch.roll(x[:, n:], -i, dims=-1)
            all_rolls[:, :, i] = torch.hstack([roll0, roll1])
        rolls_mult = x.unsqueeze(1) * all_rolls
        TC = torch.bmm(rolls_mult, all_rolls)
        TC = TC.reshape((b, k, d, d))
        return TC

    def triple_correlation_vectorized_batch_r2(self, x):
        b, k, h, w = x.shape
        x = x.reshape((b * k, h * w))
        all_rolls = torch.zeros((b * k, h * w, h * w)).to(x.device)
        for i in range(h * w):
            all_rolls[:, :, i] = torch.roll(x, -i, dims=-1)
        rolls_mult = x.unsqueeze(1) * all_rolls
        TC = torch.bmm(rolls_mult, all_rolls)
        TC = TC.reshape((b, k, h * w, h * w))
        return TC

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Apply Group Pooling to the input feature map.

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        coords = input.coords
        input = input.tensor
        b, c = input.shape[:2]
        spatial_shape = input.shape[2:]

        for s, contiguous in self._contiguous.items():
            in_indices = getattr(self, "in_indices_{}".format(s))
            out_indices = getattr(self, "out_indices_{}".format(s))

            if contiguous:
                fm = input[:, in_indices[0] : in_indices[1], ...]
            else:
                fm = input[:, in_indices, ...]

            # split the channel dimension in 2 dimensions, separating fields
            fm = fm.view(b, -1, s, *spatial_shape)

            if self.group_type == "cyclic":
                output = self.triple_correlation_vectorized_batch_cyclic(fm.squeeze())

            elif self.group_type == "dihedral":
                output = self.triple_correlation_vectorized_batch_dihedral(fm.squeeze())

            if self.idx is None:
                idx = torch.triu_indices(output.shape[2], output.shape[3])
            else:
                idx = self.idx

            output = output[:, :, idx[0], idx[1]]
            a, b, c = output.shape
            output = output.reshape((a * b, c))
            output = output / (output.norm(dim=0, keepdim=True) + 1e-5)
            output = output.reshape((a, b, c, 1, 1))
        return output

    def export(self):
        raise NotImplementedError


class TCGroupPoolingR2Spatial(torch.nn.Module):
    def __init__(self, idx=None, **kwargs):
        super().__init__(**kwargs)
        self.idx = idx

    def triple_correlation_vectorized_batch(self, x):
        b, k, n, n = x.shape
        d = n * n
        x = x.reshape((b * k, d))
        all_rolls = torch.zeros((b * k, d, d)).to(x.device)
        for i in range(d):
            all_rolls[:, :, i] = torch.roll(x, -i, dims=-1)
        rolls_mult = x.unsqueeze(1) * all_rolls
        TC = torch.bmm(rolls_mult, all_rolls)
        TC = TC.reshape((b, k, d, d))
        return TC

    def forward(self, x):
        output = self.triple_correlation_vectorized_batch(x.squeeze())
        return output

        if self.idx is None:
            idx = torch.triu_indices(output.shape[2], output.shape[3])
        else:
            idx = self.idx

        output = output[:, :, idx[0], idx[1]]
        a, b, c = output.shape
        output = output.reshape((a * b, c))
        output = output / (output.norm(dim=0, keepdim=True) + 1e-5)
        output = output.reshape((a, b, c, 1, 1))

        return output

    def export(self):
        raise NotImplementedError


class BspGroupPooling(GroupPooling):
    def __init__(self, in_type, group_type="cyclic", idx=None, **kwargs):
        """
        group_type should be "cyclic" or "dihedral"

        Parameters
        ----------
        in_type  type of geometric tensor

        """
        super().__init__(in_type, **kwargs)
        self.idx = idx
        self.group_type = group_type
    def fourier_transform_cyclic(self, f):
        """
        Computes the 1d DFT of a signal f:Z/nZ->C. Returns fhat:Z/nZ->C.
        """
        n = len(f)
        fhat = torch.zeros(n) * 1j
        for i in range(n):
            for j in range(n):
                fhat[i] += f[j] * np.exp(2 * np.pi * 1j * j * i / n)
        return fhat
        
    def fourier_transform_vectorized_batch_cyclic(self, f):
        """
        Computes the 1d DFT of a signal f:Z/nZ->C. Returns fhat:Z/nZ->C.
        """

        fc = torch.zeros(f.shape) * 1j
        fc.real = f
        n = f.shape[2]
        i_range = torch.arange(n)
        j_range = torch.arange(n)

        # Create omega tensor
        omega = 2 * toch.pi * i_range[:, None] * j_range / n * 1j
        rho = torch.tensor(torch.exp(omega))
        fhat = torch.sum(fc[:, :, :, None] * rho[None, None, :, :], axis = 2)
        return fhat

    def fourier_transform_product_cyclic(self, f):
        """
        Computes the G-DFT of a signal f:G->C where G is the sum of finitely many cyclic groups.
        Input: a signal f:G->C.
        Returns fhat:G->C.
        """
        N = len(f)
        L = len(N)
        fhat = torch.zeros(N) * 1j
        for i in np.ndindex(N):
            for j in np.ndindex(N):
                rho = 1
                for l in range(L):
                    rho *= np.exp(2 * np.pi * 1j * j[l] * i[l] / N[l])
                fhat[i] += f[j] * rho
        return fhat

    def fourier_transform_dihedral(self, f):
        """
        Input: A function f:G->C where G is the dihedral group D_n (symmetries of the n-gon).
        G = {e, a, a^2...,a^{n-1}, x, ax, a^2x,...,a^{n-1}x}.
        Output: returns the Fourier transform.
        """
        n = int(len(f)/ 2)
        n2d = int(np.floor((n - 1) / 2))
        fhat = torch.zeros(2, 2, n2d + 1)
        # the coeffs for the 1d irreps are stored in fhat[..., 0]
        # the coeffs for the 2d irreps are stored in fhat[..., 1:n2d+1 (included)]
        
        fhat[0, 0, 0] = f.sum()
        fhat[1, 0, 0] = f[:n].sum() - f[n:].sum()
        if n % 2 == 0:
            fhat[0, 1, 0] = f[0:2*n:2].sum() - f[1:2*n:2].sum()
            fhat[1, 0, 0] = f[0:n:2].sum() - f[1:n:2].sum() - f[n:2*n:2] + f[n+1:2*n:2]
        """
        for j in range(n):
            fhat[0, 0, 0] += f[j] + f[j + n]
            fhat[1, 0, 0] += f[j] - f[j + n]
        
        if n % 2 == 0:
            for j in range(0, n, 2):
                fhat[0, 1, 0] += f[j] - f[j + 1] + f[j + n] - f[j + 1 + n]
                fhat[1, 1, 0] += f[j] - f[j + 1] - f[j + n] + f[j + 1 + n]
        """
        for i in range(1, n2d + 1):
            for j in range(n):
                omega = 2 * np.pi * i * j / n
                rho = torch.tensor(
                    [[np.cos(omega), -np.sin(omega)], [np.sin(omega), np.cos(omega)]]
                )
                rho1 = rho.clone()
                fhat[..., i] += f[j] * rho
                rho1[:, 1]    *= -1
                fhat[..., i] += f[j + n] * rho1
        return fhat
    
    def fourier_transform_vectorized_batch_dihedral(self, f):
        """
        Input: A function f:G->C where G is the dihedral group D_n (symmetries of the n-gon).
        G = {e, a, a^2...,a^{n-1}, x, ax, a^2x,...,a^{n-1}x}.
        Output: returns the Fourier transform.
        """
        n = int(f.shape[2]/ 2)
        n2d = int(np.floor((n - 1) / 2))
        fhat = torch.zeros(f.shape[0], f.shape[1], 2, 2, n2d + 1)
        # the coeffs for the 1d irreps are stored in fhat[..., 0]
        # the coeffs for the 2d irreps are stored in fhat[..., 1:n2d+1 (included)]
        fhat[:, :, 0, 0, 0] = f.sum(axis=2)
        fhat[:, :, 1, 0, 0] = f[:, :, :n].sum(axis=2) - f[:, :, n:].sum(axis=2)
        if n % 2 == 0:
            fhat[:, :, 0, 1, 0] = f[:, :, 0:2*n:2].sum(axis= 2) - f[:, :, 1:2*n:2].sum(axis=2)
            fhat[:, :, 1, 0, 0] = f[:, :, 0:n:2].sum(axis= 2) - f[:, :, 1:n:2].sum(axis=2) - f[:, :, n:2*n:2].sum(axis= 2) + f[:, :, n+1:2*n:2].sum(axis = 2)
        i_range = torch.arange(1, n2d + 1)
        j_range = torch.arange(n)

        # Create omega tensor
        omega = 2 * torch.pi * i_range[:, None] * j_range / n
        # Create rho tensor
        rho = torch.concat((torch.cos(omega),-torch.sin(omega),torch.sin(omega),torch.cos(omega)))
        rho = einops.rearrange(rho, '(c1 c2 w) h  -> c1 c2 w h', c1=2, c2=2)
        rho1 = rho.clone()
        rho1[:, 1] *= -1
        fhat[..., 1:n2d+1] = torch.sum(f[:, :, None, None, None, j_range] * rho[None, None, :, :, :], dim = 5)
        fhat[..., 1:n2d+1] += torch.sum(f[:, :, None, None, None, j_range + n] * rho1[None, None, :, :, :], dim = 5)
        return fhat

    def _bispectrum_1d(self, fhat):
        """
        Compute bispectrum beta using 1d DFT.
        Input: The 1d Fourier transform on Z/nZ.
        Returns: Only the |G| bispectrum elements needed for completeness.
        These are given by beta[0,0], beta[0, 1] and beta[1, i-1] for i \in {1,2,...,n-2}
        """
        n = len(fhat)
        beta = torch.zeros(n) * 1j
        beta[0] = fhat[0] * fhat[0] * torch.conj(fhat[0])  # beta[0, 0]
        beta[1] = fhat[0] * fhat[1] * torch.conj(fhat[1])  # beta[0, 1]
        for i in range(2, n):
            beta[i] = fhat[1] * fhat[i - 1] * torch.conj(fhat[i])  # beta[1, i-1]
        return beta

    def _bispectrum_nd(self, fhat, beta):
        """
        In-place call of bispectrum_nd(fhat).
        """
        N = fhat.shape
        Nsub = tuple(list(N)[:-1])
        L = len(N)
        if L == 1:
            beta[:] = self._bispectrum_1d(fhat)
            return beta
        # Recursive call to solve the case of dimension L - 1
        beta[..., 0] = self._bispectrum_nd(fhat[..., 0], beta[..., 0])
        el = [0] * L
        nl = N[-1]
        el[L - 1] = 1
        elt1 = tuple(el)
        beta[elt1] = fhat[tuple([0] * L)] * fhat[elt1] * torch.conj(fhat[elt1])
        elt = elt1
        for i in range(1, nl):
            oldelt = elt
            el[L - 1] = i
            elt = tuple(el)
            if i > 1:
                beta[elt] = fhat[elt1] * fhat[oldelt] * torch.conj(fhat[elt])
            for k in np.ndindex(Nsub):
                if sum(k) > 0:
                    kp = list(k)
                    k = tuple(kp + [0])
                    ki = tuple(kp + [i])
                    beta[ki] = fhat[k] * fhat[elt] * torch.conj(fhat[ki])
        return beta

    def bispectrum_product_cyclic(self, x):
        """
        Input: The Fourier transform fhat defined on a commutative group G.
        Returns:  The |G| bispectral coefficients needed to obtain a complete transformation.
        """
        fhat = self.fourier_transform_product_cyclic(x)
        N = fhat.shape
        beta = torch.zeros(N) * 1j
        return self._bispectrum_nd(fhat, beta)

    def bispectrum_cyclic(self, x):
        """
        Compute bispectrum beta using 1d DFT.
        Input: The 1d Fourier transform on Z/nZ.
        Returns: Only the |G| bispectrum elements needed for completeness.
        These are given by beta[0,0], beta[0, 1] and beta[1, i-1] for i \in {1,2,...,n-2}
        """
        fhat = self.fourier_transform_cyclic(x)
        n = len(fhat)
        beta = torch.zeros(n) * 1j
        beta[0] = fhat[0] * fhat[0] * torch.conj(fhat[0])  # beta[0, 0]
        beta[1] = fhat[0] * fhat[1] * torch.conj(fhat[1])  # beta[0, 1]
        for i in range(2, n):
            beta[i] = fhat[1] * fhat[i - 1] * torch.conj(fhat[i])  # beta[1, i-1]
        return beta
    
    def bispectrum_vectorized_batch_cyclic(self, x):
        """
        Compute bispectrum beta using 1d DFT.
        Input: The 1d Fourier transform on Z/nZ.
        Returns: Only the |G| bispectrum elements needed for completeness.
        These are given by beta[0,0], beta[0, 1] and beta[1, i-1] for i \in {1,2,...,n-2}
        """
        fhat = self.fourier_transform_vectorized_batch_cyclic(x)
        b, c, n = fhat.shape
        beta = torch.zeros(fhat.shape) * 1j
        beta[..., 0] = fhat[..., 0] * fhat[..., 0] * torch.conj(fhat[..., 0])  # beta[0, 0]
        beta[..., 1] = fhat[..., 0] * fhat[..., 1] * torch.conj(fhat[..., 1])  # beta[0, 1]
        beta[..., 2:] = fhat[..., 1].unsqueeze(2) * fhat[...,1:n-1] * torch.conj(fhat[..., 2:n])
        betareal = torch.zeros(b, c, 2 * n)
        betareal[..., 0:2*n:2] = beta.real
        betareal[..., 1:2*n:2] = beta.imag
        return betareal
    
    def bispectrum_vectorized_batch_dihedral(self, x, n):
        """
        Input: Fourier transform over D_n
        Output:  the bispectral elements needed for completeness
        """
        fhat = self.fourier_transform_vectorized_batch_dihedral(x)
        
        bs, cs = fhat.shape[:2]
        #computes beta_\rho_0,\rho_0
        beta0 = fhat[:, :, 0, 0, 0] ** 3
        #computes beta_\rho_1,\rho_1
        beta10 =  torch.sum(fhat[:, :, None, None, None, 0, 0, 0] * fhat[:, :, :, None, :, 1] * fhat[:, :, None, :, :, 1], axis = 4).squeeze()

        n2 = int(np.floor((n - 1) / 2))
        n3 = n2
        if n % 2 > 0:
            n3 = n2 - 1
        beta1i = torch.zeros(bs, cs, 4, 4, n3)
        indices = np.zeros(2, dtype = int)
        
        CBmatrix, indices = first_last_cb(n, end=False)
        Fplus = build_Fplus_vectorized(indices, fhat, n, end=False)
        #beta = (fhat \otimes fhat) * C * F.T * C.T = (fhat \otimes fhat) * C * (C * F).T
        fh_kron_fh = (fhat[:, :, :, None, :, None, 1] * fhat[:, :, None, :, None, :, 1]).reshape((bs, cs, 4, 4))
        fh_kron_fh_C = torch.sum(fh_kron_fh[:, :, :, :, None] * CBmatrix[None, None, None, :, :], axis = 3)
        C_Fplus = torch.sum(CBmatrix[None, None, :, :, None] * Fplus[:, :, None, :, :], axis = 3)
        beta1i[..., 0] = torch.sum(fh_kron_fh_C[:, :, :, None, :] * C_Fplus[:, :, None, :, :], axis = 4)
        Fplus = torch.zeros(bs, cs, 4, 4)
        CBmatrix = CBmatrix.clone()
        
        for i in range(2, n2):
            CBmatrix, indices = clebsch_gordan(1, i, n)
            Fplus[..., :2, :2] = fhat[..., indices[0]]
            Fplus[..., 2:, 2:] = fhat[..., indices[1]]
            fh_kron_fh = (fhat[:, :, :, None, :, None, 1] * fhat[:, :, None, :, None, :, i]).reshape((bs, cs, 4, 4))
            fh_kron_fh_C = torch.sum(fh_kron_fh[:, :, :, :, None] * CBmatrix[None, None, None, :, :], axis = 3)
            C_Fplus = torch.sum( CBmatrix[None, None, :, :, None] * Fplus[:, :, None, :, :], axis = 3)
            beta1i[..., i - 1] = torch.sum( fh_kron_fh_C[:, :, :, None, :] * C_Fplus[:, :, None, :, :], axis = 4)
  
        Fplus = torch.zeros(bs, cs, 4, 4)
        CBmatrix = CBmatrix.clone()
        if n % 2 == 0:
            CBmatrix, indices = first_last_cb(n, end=True)
            Fplus = build_Fplus_vectorized(indices, fhat, n, end=True)
            fh_kron_fh = (fhat[:, :, :, None, :, None, 1] * fhat[:, :, None, :, None, :, n2]).reshape((bs, cs, 4, 4))
            fh_kron_fh_C = torch.sum(fh_kron_fh[:, :, :, :, None] * CBmatrix[None, None, None, :, :], axis = 3)
            C_Fplus = torch.sum(CBmatrix[None, None, :, :, None] * Fplus[:, :, None, :, :], axis = 3)
            beta1i[..., n2 - 1] = torch.sum( fh_kron_fh_C[:, :, :, None, :] * C_Fplus[:, :, None, :, :], axis = 4)
        
        beta0 = beta0.unsqueeze(2)
        bs, cs, a, b = beta10.shape
        beta10 = beta10.reshape((bs, cs, a * b))
        bs, cs, a, b, c = beta1i.shape
        beta1i = beta1i.reshape((bs, cs, a * b * c))
        return torch.cat((beta0, beta10, beta1i), dim = 2)
    
    def bispectrum_dihedral(self, x, n):
        """
        Input: Fourier transform over D_n
        Output:  the bispectral elements needed for completeness
        """
        fhat = self.fourier_transform_dihedral(x)
        #return torch.ravel(fhat)
        #computes beta_\rho_0,\rho_0
        beta0 = torch.ones(1) * fhat[0, 0, 0] ** 3
        #computes beta_\rho_1,\rho_1
        beta10 = fhat[0, 0, 0] * (fhat[..., 1] @ fhat[..., 1].T)
        n2 = int(np.floor((n - 1) / 2))
        n3 = n2
        if n % 2 > 0:
            n3 = n2 - 1
        beta1i = torch.zeros(4, 4, n3)
        indices = np.zeros((2, n3), dtype = int)
        CBmatrices = torch.zeros(4, 4, n3)
            
        CBmatrices[..., 0], indices[:, 0] = first_last_cb(n, end=False)
        Fplus = buildFplus(indices[:, 0], fhat, n, end=False)
        beta1i[..., 0] = torch.kron(fhat[..., 1], fhat[..., 1]) @ CBmatrices[..., 0] @ Fplus.T @ (CBmatrices[..., 0].T)
        Fplus = torch.zeros(4, 4)
        CBmatrices = CBmatrices.clone()
        for i in range(2, n2):
            CBmatrices[..., i - 1], indices[:, i - 1] = clebsch_gordan(1, i, n)
            Fplus[:2, :2] = fhat[..., indices[0, i - 1]]
            Fplus[2:, 2:] = fhat[..., indices[1, i - 1]]
            beta1i[..., i - 1] = torch.kron(fhat[..., 1], fhat[..., i]) @ CBmatrices[..., i - 1] @ Fplus.T @ (CBmatrices[..., i - 1].T)
        Fplus = torch.zeros(4, 4)
        CBmatrices = CBmatrices.clone()
        if n % 2 == 0:
            CBmatrices[..., n2 - 1], indices[:, n2 - 1] = first_last_cb(n, end=True)
            Fplus = buildFplus(indices[:, n2 - 1], fhat, n, end=True)
            beta1i[..., n2 - 1] = torch.kron(fhat[..., 1], fhat[..., n2]) @ CBmatrices[..., n2 - 1] @ Fplus.T @ (CBmatrices[..., n2 - 1].T)
        
        return torch.cat((beta0,torch.ravel(beta10),torch.ravel(beta1i)))#(beta0, beta10, beta1i)#, CBmatrices, indices
    

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        """

        Apply Group Pooling to the input feature map.

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        coords = input.coords
        input = input.tensor
        b, c = input.shape[:2]  # b = batch size, c = channel size
        spatial_shape = input.shape[2:]

        for s, contiguous in self._contiguous.items():
            in_indices = getattr(self, "in_indices_{}".format(s))  # self.in_indices_0
            out_indices = getattr(self, "out_indices_{}".format(s))

            if contiguous:
                fm = input[:, in_indices[0] : in_indices[1], ...]
            else:
                fm = input[:, in_indices, ...]

            # split the channel dimension in 2 dimensions, separating fields
            fm = fm.view(b, -1, s, *spatial_shape)
            # fm is feature map
            bm, cm = fm.shape[:2]
            if self.group_type == "cyclic":
                n = fm.shape[2]
                output = self.bispectrum_vectorized_batch_cyclic(fm.squeeze())
            elif self.group_type == "product_cyclic":
                output = self.bispectrum_product_cyclic(fm.squeeze())
            elif self.group_type == "dihedral":
                n = int(fm.shape[2] / 2)
                output = self.bispectrum_vectorized_batch_dihedral(fm.squeeze(), n)

            """
            if self.idx is None:
                idx = torch.triu_indices(output.shape[2], output.shape[3])
            else:
                idx = self.idx
            """
            #output = output[:, :, idx[0], idx[1]]
            a, b, c = output.shape
            output = output.reshape((a * b, c))
            output = output / (output.norm(dim=0, keepdim=True) + 1e-5)
            output = output.reshape((a, b, c, 1, 1))
        return output

    def export(self):
        raise NotImplementedError
