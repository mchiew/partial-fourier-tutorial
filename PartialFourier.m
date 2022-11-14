%% Partial Fourier
% mark.chiew@ndcn.ox.ac.uk
% 
% A brief overview of some Partial Fourier MR Image Reconstruction methods. 
% 
% References:
% 
% * Noll DC, Nishimura DG, Macovski A. Homodyne detection in magnetic resonance 
% imaging. IEEE Trans Med Imaging 1991;10:154?163. doi: 10.1109/42.79473.
% * Haacke EM, Lindskogj ED, Lin W. A fast, iterative, partial-fourier technique 
% capable of local phase recovery. Journal of Magnetic Resonance (1969) 1991;92:126?145.
% * Liang Z-P, Boda FE, Constable RT, Haacke EM, Lauterbur PC, Smith MR. Constrained 
% Reconstruction Methods in MR Imaging. Reviews of Magnetic Resonance in Medicine 
% 1992;4:67?185.
% * McGibney G, Smith MR, Nichols ST, Crawley A. Quantitative evaluation of 
% several partial Fourier reconstruction algorithms used in MRI. Magn Reson Med 
% 1993;30:51?59.
% * Willig-Onwuachi JD, Yeh EN, Grant AK, Ohliger MA, McKenzie CA, Sodickson 
% DK. Phase-constrained parallel MR image reconstruction. Journal of Magnetic 
% REsonance (1969) 2005;176:187?198. doi: 10.1016/j.jmr.2005.06.004.
% * Blaimer M, Gutberlet M, Kellman P, Breuer FA, Köstler H, Griswold MA. Virtual 
% coil concept for improved parallel MRI employing conjugate symmetric signals. 
% Magn Reson Med 2009;61:93?102. doi: 10.1002/mrm.21652.
% * Haacke EM, Brown RW, Thompson MR, Venkatesan R. Magnetic resonance imaging: 
% physical principles and sequence design. Wiley-Liss; 1999.
% * Bernstein MA, King KF, Zhou XJ. Handbook of MRI Pulse Sequences. Academic 
% Press; 2004.
%% Load Data
%%
% synthetic data
x = (1:96); y = (1:96)';
I_mag = phantom(96);
I_phs = 2*pi*(3*exp(-sqrt((x-36).^2+(y-36).^2)/16)+2*exp(-((x-56).^2)/36^2));
img = I_mag.*exp(1j*I_phs);

% real data
% can be downloaded from https://users.fmrib.ox.ac.uk/~mchiew/docs/pf_data.mat
%load('pf_data.mat');
%% Show Data
%%
% magnitude on the left, phase on the right
show_pair(abs(img),[0 1],angle(img).*(abs(img)>1E-2),[-pi pi])
%% Sample Data and Generate a Low-Resolution Phase Estimate
% We simulate 6/8 or 3/4 Partial Fourier sampling by cutting off the bottom 
% 25% of k-space

% 3/4 Partial Fourier Sampling
imgPF = fftdim(img,1:2);
imgPF(73:96,:) = 0;
clf; imshow(log10(abs(imgPF)),[]); title('3/4 Partial Fourier Sampling')
%% 
% From the symmetrically sampled central k-space region, we can generate 
% a lower resolution estimate of the image phase

% Low resolution phase estimate from central k-space
phs = zeros(96);
phs(25:72,:) = imgPF(25:72,:).*hann(48);
phs = exp(1j*angle(ifftdim(phs,1:2)));
show_pair(angle(img).*(abs(img)>1E-2),[-pi,pi], angle(phs).*(abs(img)>1E-2),[-pi,pi])
%% Algorithm 1 - Zero Filling
% Zero filling is the most basic Partial Fourier "reconstruction" - it just 
% sets un-sampled k-space locations as zero, and performs the conventional inverse 
% Fourier transform.

% direct zero-filling reconstruction
img_zf = abs(ifftdim(imgPF,1:2));
show_pair(img_zf,[0 1],10*img_diff(img_zf,img),[0 1])
nrmse(img_zf,img)
%% 
% To prevent Gibbs ringing, k-space may be apodized/filtered prior to reconstruction 
% to more smoothly transition from sampled k-space to zero.

% zero-filling with apodizing filter
W = 1./(1+exp((-66:29)'/1));
plt(W,'k-space filter');
img_zfw = abs(ifftdim(imgPF.*W,1:2));
show_pair(img_zfw,[0 1],10*img_diff(img_zfw,img),[0 1])
nrmse(img_zfw,img)
%% Algorithm 2 - Conjugate Synthesis
% One defining feature of Partial Fourier reconstructions is the use of the 
% fact that when your image is purely real (or equivalently, you know the image 
% phase), k-space has conjugate symmetry. That is, 
% 
% $$S\left(-k\right)=S^* \left(k\right)$$
% 
% where $S\left(k\right)$ is the sampled signal at k-location $k$, and $S^* 
% \left(k\right)$ is its complex conjugate.
% 
% In 1D, it would look something like: $\left\lbrack \begin{array}{ccccccccc}A 
% & B & C & D & 0 & D^*  & C^*  & B^*  & \end{array}\right\rbrack$
% 
% To get the missing entry, we can simply look at the value in the mirror-symmetric 
% position opposite the origin ($A$), and conjugate its value to get ($A^*$). 
% In most cases, Partial Fourier is performed along one axis, and the other axis 
% can be fully reconstructed via inverse Fourier Transform, which results in a 
% bunch of independent 1D problems anyway.
% 
% Let's see what happens when we naively use the conjugate symmetry principle 
% on data that doesn't actually satisfy the constraint of being purely real:

img_conj = ifftdim(imgPF,2);
img_conj(73:end,:) = flip(conj(img_conj(2:25,:)),1);
img_conj = abs(ifftdim(img_conj,1));
show_pair(img_conj,[0 1],10*img_diff(img_conj,img),[0 1])
nrmse(img_conj,abs(img))
%% Algorithm 3 - Homodyne/Margosian
% Without ensuring the image is real, synthesizing conjugate signals from the 
% symmetric locations in k-space clearly produces poor results. In fact, conjugate 
% synthesis without using the phase results in worse reconstruction that zero-filling!
% 
% To illustrate the importance of the phase correction, cosider the visible 
% up/down symmetry in the phase corrected k-space (right) compared to the original 
% sampled k-space (left):

show_pair(abs(ifftdim(imgPF,2)),[],abs(fftdim(ifftdim(imgPF,1:2).*conj(phs),1)),[]);
%% 
% In the Homodyne/Margosian approach, we exploit conjugate symmetry along 
% with our knowledge of the image phase to produce a more robust estimate of the 
% image. The approach filters the Partial Fourier sampled k-space, before inverse 
% Fourier transform and phase alignment, resulting in a real-valued final output.
% 
% To understand why, lets consider our k-space split into high and low-frequency 
% segments: 
% 
% $$S\left(k\right)=\left\lbrack \begin{array}{ccc}H\left(k\right) & L\left(k\right) 
% & H\left(k\right)\end{array}\right\rbrack$$ 
% 
% where $H$ represents both (positive and negative) high frequency k-spaces, 
% and $L$ is the low-frequency centre of k-space in between. Because the Fourier 
% transform is linear, we can consider our image as:
% 
% $$I\left(x\right)e^{i\phi } =F^{-1} \left(S\right)=F^{-1} \left(H+L\right)=F^{-1} 
% \left(H\right)+F^{-1} \left(L\right)=h\left(x\right)e^{i\phi } +l\left(x\right)e^{i\phi 
% }$$
% 
% However, in a Partial Fourier acquisition, we only have one half of the 
% high frequencies, so that
% 
% $$S_{\mathrm{PF}} \left(k\right)=\left\lbrack \begin{array}{ccc}0 & L\left(k\right) 
% & H\left(k\right)\end{array}\right\rbrack =u\;H+L$$ 
% 
% $$I_{\mathrm{PF}} \left(x\right)=F^{-1} \left(u\;H+L\right)=F^{-1} \left(u\;H\right)+F^{-1} 
% \left(L\right)=h\left(x\right)e^{i\phi } \ast \frac{1}{2}\left(\delta +\frac{1}{i\pi 
% x}\right)+l\left(x\right)e^{i\phi }$$
% 
% where $u$ is the Heavyside step function, which zeros out the negative 
% high frequency data, leaving only the positive frequency half of $H$, and where 
% inverse Fourier Transform of $u\left(k\right)$ is $\frac{1}{2}\left(\delta +\frac{1}{i\pi 
% x}\right)$.
% 
% So, because the high spatial frequencies are only weighted by |1/2|, we 
% up-weight the remaining high-frequency component to compensate, to get:
% 
% $$\begin{array}{l}I_0 \left(x\right)=F^{-1} \left(u\;2H+L\right)=F^{-1} 
% \left(u\;2H\right)+F^{-1} \left(L\right)=h\left(x\right)e^{i\phi } \ast \left(\delta 
% +\frac{1}{i\pi x}\right)+l\left(x\right)e^{i\phi } =h\left(x\right)e^{i\phi 
% } +l\left(x\right)e^{i\phi } +h\left(x\right)e^{i\phi } \ast \frac{1}{i\pi x}\\I_0 
% \left(x\right)\approx I\left(x\right)e^{i\phi } +\left\lbrack h\left(x\right)\ast 
% \frac{1}{i\pi x}\right\rbrack e^{i\phi } \end{array}$$
% 
% Now we can see from the above that by upweighting the high-frequency component 
% $H$, the desired output $I\left(x\right)$ is now in the expression for $I_0 
% \left(x\right)$, along with a convolution nuisance term. The expression for 
% $I_0 \left(x\right)$ is now approximate because the phase term was pulled out 
% of the convolution, which is an approximation that only holds for slowly varying 
% phase.
% 
% Finally, we can extract our reconstructed image by taking the real part 
% of the phase-corrected $I_0 \left(x\right)$:
% 
% $$\begin{array}{l}I_{\mathrm{Final}} \left(x\right)=\mathrm{Re}\left\lbrack 
% I_0 \left(x\right)e^{-i\phi } \right\rbrack \\I_{\mathrm{Final}} \left(x\right)\approx 
% \mathrm{Re}\left\lbrack \left(I\left(x\right)e^{i\phi } +\left\lbrack h\left(x\right)\ast 
% \frac{1}{i\pi x}\right\rbrack e^{i\phi } \right)e^{-i\phi } \right\rbrack =I\left(x\right)\end{array}$$
% 
% The phase correction makes $I\left(x\right)$ purely real, and the nuisance 
% term $h\left(x\right)\ast \frac{1}{i\pi x}$ purely imaginary, so that $I\left(x\right)$ 
% can be extracting by taking the real part.
% 
% So, to summarize, the reconstruction is simply a matter of:
% 
% # Re-weighting the sampled k-space (no explicit conjugating or filling necessary)
% # Inverse Fourier transforming the re-weighted k-space
% # Phase correcting the result
% # Taking the real part of the phase-corrected image as the output

% generate re-weighting vector
filt = zeros(96,1);
filt(26:72) = 1;    % L
filt(1:25) = 2;     % H
plt(filt,'Homodyne Filter')
img_homodyne = real_nonneg(conj(phs).*ifftdim(imgPF.*filt,1:2));
show_pair(img_homodyne,[0 1],10*img_diff(img_homodyne,img),[0 1])
nrmse(img_homodyne,img)
%% 
% In fact, the k-space re-weighting can be more generally thought of as 
% a reconstruction spatial filter, and we can take advantage of that to produce 
% a more smoothly varying re-weighting to reduce Gibbs ringing/truncation artefacts:

filt2 = smoothdata(filt,1,'gaussian',15);
plt(filt2,'Smoothed Homodyne Filter')
img_homodyne2 = real_nonneg(conj(phs).*ifftdim(ifftdim(imgPF,2).*filt2,1));
show_pair(img_homodyne2,[0 1],10*img_diff(img_homodyne2,img),[0 1])
nrmse(img_homodyne2,img)
%% Algorithm 4 - Projection onto convex sets (POCS)
% Now we consider a different approach to Partial Fourier reconstruction, using 
% an iterative reconstruction method called "Projection onto Convex Sets", or 
% POCS. To do this, let's consider what information we have to solve our under-determined 
% image reconstruction problem:
% 
% # We know the data should be consistent with the measured 75% of k-space
% # We know the image phase, or equivalently, we know the our estimate should 
% be non-negative real
% 
% and can define the solution to our image reconstruction problem as any 
% image where both |1| and |2| are true.
% 
% If we redefine |1| and |2 |as set constraints, we can then re-state our 
% solution as any image that lies in the intersection of these two constraint 
% sets. POCS, it turns out, is a general algorithm for finding set intersections, 
% assuming the constraint sets are convex (which they are in this case).
% 
% The way POCS works is simply by alternating between projecting the current 
% iterate onto each constraint set in turn, until the iterate stops changing, 
% which means the intersection has been reached. Here, we alternate between projecting 
% onto the data-consistency set $P_1$ (by replacing the sampled k-space into the 
% currrent iterate), and projecting onto the set of non-negative real numbers 
% $P_2$ (by setting each voxel to be real and ? 0):
% 
% $$I^{n+1} \left(x\right)=$$$$P_2 \left(P_1 \left(I^n \left(x\right)\right)\right)$$

% Initialise parameters, starting image of zeros
ref = ifftdim(imgPF,2);
img_pocs = zeros(96);
diff = inf;
iter = 0;

while diff > 1E-6 && iter < 10
    % Projection onto data-consistency set
    tmp = img_pocs.*phs;
    tmp = fftdim(tmp,1);
    tmp(1:72,:) = ref(1:72,:);
    tmp = ifftdim(tmp,1).*conj(phs);
    
    % Projection onto non-negative real set
    tmp = real_nonneg(tmp);
    
    % update 
    diff = norm(tmp(:) - img_pocs(:))/norm(img_pocs);   
    img_pocs = tmp;
    iter = iter + 1;
end

show_pair(img_pocs,[0 1],10*img_diff(img_pocs,img),[0 1])
nrmse(img_pocs,abs(img))
%% Algorithm 5 - Phase Constrained Reconstruction and the Virtual Coil Concept
% If we consider the problem now just as a general linear inverse problem, we 
% can solve our Partial Fourier reconstruction using similar tools and formulations 
% to those used in Parallel Imaging. In fact, this is precisely one way to combine 
% Partial Fourier/Phase Constraints with Parallel Imaging, if coil sensitivity 
% information is present and being used.
% 
% Let's consider our signal again:
% 
% $$S\left(k\right)=F\left(I\left(x\right)e^{i\phi } \right)$$
% 
% Because our phase term $e^{i\phi }$ is known, lets actually just pretend 
% our phase term is a complex valued coil sensitivity applied to our real image:
% 
% $S\left(k\right)=F\left(C_1 \left(x\right)I\left(x\right)\right)$, where 
% $C_1 \left(x\right)=e^{i\phi \left(x\right)}$
% 
% Now, let's consider the conjugate symmetric signal:
% 
% $S^* \left(-k\right)=F\left(I\left(x\right)e^{-i\phi } \right)=F\left(C_2 
% \left(x\right)I\left(x\right)\right)$, where $C_2 \left(x\right)=e^{-i\phi \left(x\right)}$
% 
% Now, if we collect all of our information into a single linear system we 
% get:
% 
% $$\left\lbrack \begin{array}{c}S\left(k\right)\\S^* \left(-k\right)\end{array}\right\rbrack 
% =\left\lbrack \begin{array}{cc}F & 0\\0 & F\end{array}\right\rbrack \left\lbrack 
% \begin{array}{c}C_1 \\C_2 \end{array}\right\rbrack I$$
% 
% we can then solve a SENSE-like problem with virtual coils $C$ as:
% 
% $$\left\lbrack \begin{array}{c}F^{-1} \left(S\left(k\right)\right)\\F^{-1} 
% \left(S^* \left(-k\right)\right)\end{array}\right\rbrack =\left\lbrack \begin{array}{c}C_1 
% \\C_2 \end{array}\right\rbrack =C\;I$$
% 
% which has a least squares solution:
% 
% $$\hat{I} ={\left(C^* C\right)}^{-1} C^* \left\lbrack \begin{array}{c}F^{-1} 
% \left(S\left(k\right)\right)\\F^{-1} \left(S^* \left(-k\right)\right)\end{array}\right\rbrack$$

% define virtual coil matrix
C = [spdiags(phs(:),0,96^2,96^2);spdiags(conj(phs(:)),0,96^2,96^2)];
%% 
% 

% compute inverse Fourier transforms of sampled data
A = ifftdim(imgPF,1:2);
B = ifftdim(conj(circshift(flip(circshift(flip(imgPF,1),1,1),2),1,2)),1:2);
%% 
% 

% solve
img_vc = inv(C'*C)*C'*[A(:);B(:)];
img_vc = reshape(real_nonneg(img_vc),96,96);
show_pair(img_vc,[0 1],10*img_diff(img_vc,img),[0 1])
nrmse(img_vc,img)
% solve iteratively with (too much) regularisation
e = ones(96*96,1);
R = spdiags([-e 2*e -e], -1:1, 96*96, 96*96);
img_vc2 = pcg(C'*C+1E0*R, C'*[A(:);B(:)],1E-9,100);
img_vc2 = reshape(real_nonneg(img_vc2),96,96);
show_pair(img_vc2,[0 1],10*img_diff(img_vc2,img),[0 1])
nrmse(img_vc2,img)
%% Helper Functions
%%
function out = nrmse(a,b)
    out = norm(a(:)-abs(b(:)))/norm(abs(b(:)));
end

function show_pair(dataL, cscaleL, dataR, cscaleR)
    clf();
    subplot('position',[0   0   0.5 1]);imshow(dataL,cscaleL)
    subplot('position',[0.5 0   0.5 1]);imshow(dataR,cscaleR)
end
function plt(data,label)
    clf();
    plot(data,'linewidth',2);
    grid on;
    title(label);
end

function c = img_diff(a,b)
    c = abs(a - abs(b));
end

function x = fftdim(x,dims)
    for i = dims
        x = fftshift(fft(ifftshift(x,i),[],i),i);
    end
end
function x = ifftdim(x,dims)
    for i = dims
        x = fftshift(ifft(ifftshift(x,i),[],i),i);
    end
end

function w = hann(N)
    w = 0.5*(1-cos(2*pi*(0:N-1)'/(N-1)));
end

function b = real_nonneg(a)
    b = max(real(a),0);
end