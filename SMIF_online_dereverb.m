function [s_mc,s_pw] = SMIF_online_dereverb(soundData,cctfLen)
%% Input
% SoundData : multichannel recording, with sampling rate of 16,000 Hz
% cctfLen: length of critically sampled ctf. Important! it is approximately set to
% reverberation time divided by 48 ms, note here reverberation time is not
% T60, instead it is the time duration that covers the main
% dereverberation, closely as T20 (T60/3).
%
%% Output
% s_mc: dereverberated signal based on multichannel scheme
% s_pw: dereverberated signal based on pairwise scheme
% Normally, the second performs slightly better than the first one, the
% user can use either of them

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is an online implementation for multichannel dereverberation
%
% The method is described in the paper:
%
% - X. Li, L. Girin, S. Gannot and R. Horaud. Multichannel Online Dereverberation based on Spectral Magnitude Inverse Filtering. TASLP 2019.
%
% Author: Xiaofei Li, INRIA Grenoble Rhone-Alpes
% Copyright: Perception Team, INRIA Grenoble Rhone-Alpes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<2
    cctfLen = 3;
end

if size(soundData,1)<size(soundData,2)
    soundData = soundData';
end
micNum = size(soundData,2);

%% STFT parameters
fs = 16000;
winLen = 768;                            % frame length
fraInc = winLen/4;                       % frame step
fstp = winLen/fraInc;                    % 4, decimation factor from oversampled to critically sampled
win = hanning(winLen);                   % window

freRan = 1:winLen/2+1;
freNum = length(freRan);

winCoe = zeros(fraInc,1);
for i = 1:fstp
    winCoe = winCoe+win((i-1)*fraInc+1:i*fraInc).^2;
end
winCoe = repmat(winCoe,[fstp,1]);
awin = win./sqrt(winCoe*winLen);        % analysis window
swin = win./sqrt(winCoe/winLen);        % synthesis window

fraNum = floor((size(soundData,1)-winLen)/fraInc);         % number of frames

%% CTF identification parameters

octfLen = (cctfLen-1)*fstp+1;            % length of oversampled ctf
ftHis = zeros(freNum,micNum,octfLen);    % STFT coefficients with length of oversampled ctf for ctf identification and inverse filtering

vctfLen = cctfLen*micNum;                % length of ctf vector

covInv = zeros(freNum,vctfLen,vctfLen);  % inverse of signal covariance matrix
for fre = 1:freNum                       % initialization
    covInv(fre,:,:) = 1e3*eye(vctfLen);
end

fraNumLs = 10*cctfLen;                   % memory length for RLS
lambda = (fraNumLs-1)/(fraNumLs+1);

consVec = zeros(vctfLen,1);              % constraint vector, set the summation of the first entry of all channels to 1
consVec(1:cctfLen:end) = 1;

%% Inverse Filtering Parameters
ifiLen = cctfLen;                        % length of inverse filter

tarFunLen = cctfLen+ifiLen-1;
tarFun = zeros(tarFunLen,1);             % target function of inverse filtering, impulse function
tarFun(1) = 1;

Nfft = 2^(nextpow2(tarFunLen)+1);        % fft length for gradient calculation

mu = 0.05;                               % gradient descend step-size

snrPriMin = 10^(-15/10);                 % lower limit gain factor
gainLim = snrPriMin^0.5;

%% Multichannel inverse filtering
ifiLmsMc = zeros(freNum,micNum,ifiLen);     % inverse filters
s_mc = zeros((fraNum-1)*fraInc+winLen,1);   % dereverberated signal

%% Pairwise inverse filtering
mpNum = micNum*(micNum-1)/2;
MP = [];                                    % microphone pair
for m1 = 1:micNum-1
    for m2 = m1+1:micNum
        MP = [MP;[m1,m2]];
    end
end

ifiLmsPw = zeros(freNum,mpNum,2,ifiLen);    % inverse filters
s_pw = zeros((fraNum-1)*fraInc+winLen,1);   % dereverberated signal


%% Online dereverberation
WH = waitbar(0,'Please wait ...');
for j = 1:fraNum
    
    % Short-time time-domain signal
    xt = soundData((j-1)*fraInc+1:(j-1)*fraInc+winLen,:);
    
    % FFT
    xft = fft(bsxfun(@times,xt,awin));
    xft = xft(freRan,:);
    
    % FT coefficients for ctf indentification and inverse filtering
    ftHis(:,:,2:end) = ftHis(:,:,1:end-1);
    ftHis(:,:,1) = xft;      
    
    %% RLS critically sampled ctf identification
    
    covInv = covInv/lambda;
    m1 = 1;
    M2 = 1:micNum; M2(m1) = [];
    for m2 = M2
        vecx = zeros(freNum,vctfLen);
        vecx(:,(m1-1)*cctfLen+1:m1*cctfLen) = squeeze((ftHis(:,m2,1:fstp:end)));
        vecx(:,(m2-1)*cctfLen+1:m2*cctfLen) = -squeeze((ftHis(:,m1,1:fstp:end)));
        
        Px = sum(bsxfun(@times,covInv,reshape(conj(vecx),[freNum,1,vctfLen])),3);
        Px = Px./repmat(1+sum(Px.*vecx,2),[1,vctfLen]);
        covInv = covInv-bsxfun(@times,repmat(Px,[1,1,vctfLen]),sum(bsxfun(@times,covInv,vecx),2));
    end
    
    ctfRls = sum(bsxfun(@times,covInv,reshape(consVec,[1,1,vctfLen])),3);
    ctfRls = ctfRls./repmat(sum(bsxfun(@times,reshape(consVec,[1,vctfLen]),ctfRls),2),[1,vctfLen]); % identified ctf via RLS
    
    ctfMag = permute(abs(reshape(ctfRls,[freNum,cctfLen,micNum])),[1 3 2]);          % ctf magnitude
    
    %% Multichannel inverse filtering    
    ctfMc = bsxfun(@times,ctfMag,1./sum(ctfMag(:,:,1),2));    
    
    ctfDft1 = fft(ctfMc,Nfft,3);
    ctfDft2 = conj(ctfDft1);
    
    ifiDft = fft(ifiLmsMc,Nfft,3);
    
    fiCon = real(ifft(sum(ctfDft1.*ifiDft,2),[],3));
    fiErr = bsxfun(@minus,fiCon(:,:,1:tarFunLen),reshape(tarFun,[1,1,tarFunLen]));
    fiErrDft = fft(fiErr,Nfft,3);
    
    derFi = real(ifft(bsxfun(@times,ctfDft2,fiErrDft),[],3));
    derFi = derFi(:,:,1:ifiLen);
    
    Tr = cctfLen*sum(sum(ctfMc.^2,3),2);
    
    ifiLmsMc = ifiLmsMc-bsxfun(@times,derFi, mu./Tr);            % inverse filter update 
    
    % inverse filtering
    sMag = sum(sum(abs(ftHis(:,:,1:fstp:end)).*ifiLmsMc,3),2);
    xMagLim = gainLim*mean(abs(xft),2);
    sMag =  sMag.*(sMag>xMagLim) + xMagLim.*(sMag<=xMagLim);
    sft = sMag.*exp(1i*angle(xft(:,1)));
    S(:,j) = sft;
    sftFB = [sft;conj(sft(end-1:-1:2))];
    s_mc((j-1)*fraInc+1:(j-1)*fraInc+winLen) = s_mc((j-1)*fraInc+1:(j-1)*fraInc+winLen)+swin.*real(ifft(sftFB));
    
   
    %% Pairwise inverse filtering    
    
    ctfPw = zeros(freNum,mpNum,2,cctfLen);
    for mp = 1:mpNum
        ctfPw(:,mp,:,:) = bsxfun(@times,ctfMag(:,MP(mp,:),:),1./sum(ctfMag(:,MP(mp,:),1),2));
    end
    
    ctfDft1 = fft(ctfPw,Nfft,4);
    ctfDft2 = conj(ctfDft1);
    
    ifiDft = fft(ifiLmsPw,Nfft,4);
    
    fiCon = real(ifft(sum(ctfDft1.*ifiDft,3),[],4));
    fiErr = bsxfun(@minus,fiCon(:,:,:,1:tarFunLen),reshape(tarFun,[1,1,1,tarFunLen]));
    fiErrDft = fft(fiErr,Nfft,4);
    
    derFi = real(ifft(bsxfun(@times,ctfDft2,fiErrDft),[],4));
    derFi = derFi(:,:,:,1:ifiLen);
    
    Tr = cctfLen*sum(sum(ctfPw.^2,4),3);
    
    ifiLmsPw = ifiLmsPw-bsxfun(@times,derFi, mu./Tr);            % inverse filter update
    
    % inverse filtering
    sMag = 0;
    for mp = 1:mpNum
        sMag = sMag+sum(sum(abs(ftHis(:,MP(mp,:),1:fstp:end)).*squeeze(ifiLmsPw(:,mp,:,:)),3),2);
    end
    sMag = sMag/mpNum;
    
    sMag =  sMag.*(sMag>xMagLim) + xMagLim.*(sMag<=xMagLim);
    sft = sMag.*exp(1i*angle(xft(:,1)));
    sftFB = [sft;conj(sft(end-1:-1:2))];
    s_pw((j-1)*fraInc+1:(j-1)*fraInc+winLen) = s_pw((j-1)*fraInc+1:(j-1)*fraInc+winLen)+swin.*real(ifft(sftFB));        
     
    waitbar(j/fraNum)
end
close(WH)


