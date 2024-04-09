FROM ubuntu:20.04 AS base-ubuntu-20-04-amd64

RUN apt update
RUN apt install -y --no-install-recommends apt-utils dialog

RUN curl -L https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh | bash
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y gitlab-runner iputils-ping libboost-all-dev sudo awscli zip unzip
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y lldb zsh tmux vim emacs tree git

##################################################
# Extra Packages
##################################################
RUN apt update && apt install -y pciutils kmod python3-pip
RUN apt update && apt install -y python3.8 python3.8-dev python3.8-venv
RUN apt update && apt install -y libdpkg-perl
RUN apt update && apt install -y curl jq
RUN apt update && apt install -y cmake=3.16.3-1ubuntu1 cmake-data=3.16.3-1ubuntu1
RUN apt update && apt install -y hwloc nano
RUN apt update && apt install -y valgrind=1:3.15.0-1ubuntu9
RUN apt update && apt install -y locales

RUN pip3 install z3-solver==4.8.15
RUN pip3 install junitparser==2.5.0
RUN pip3 install elasticsearch==7.16.3
RUN pip3 install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

##################################################
# DeBuda Packages
##################################################
RUN apt update && apt install -y libzmq3-dev
RUN pip3 install pyzmq tabulate

##################################################
# Helper utilities - Grendel
##################################################
RUN apt update && apt install -y \
    build-essential \
    python3.8 \
    screen \
    tmux \
    binutils \
    automake \
    build-essential \
    libboost-dev=1.71.0.0ubuntu2 \
    wget \
    gdb=9.2-0ubuntu1~20.04.1 \
    gfortran \
    git \
    g++-9 \
    vim \
    emacs \
    sudo \
 && rm -rf /var/lib/apt/lists/*

##################################################
# For Grendel perf model
##################################################
RUN apt update && apt install -y \
    libboost-all-dev \
    libyaml-cpp-dev \
    rapidjson-dev \
    libsqlite3-dev \
    libhdf5-serial-dev \
    doxygen \
 && rm -rf /var/lib/apt/lists/*

##################################################
# Sudo
##################################################
RUN chmod o+w /etc/sudoers && echo "%linux-admins ALL= ALL" >> /etc/sudoers && chmod o-w /etc/sudoers

##################################################
# Set up SSH
##################################################
RUN apt update && apt install -y openssh-server
RUN update-rc.d ssh defaults
RUN echo "X11UseLocalhost no" >> /etc/ssh/sshd_config

##################################################
# Set up Active Directory
##################################################
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y realmd policykit-1 libnss-sss libpam-sss adcli packagekit samba chrony krb5-user sssd sssd-tools
RUN systemctl enable sssd

##################################################
# Build Environment
##################################################
RUN apt update && apt install -y libtbb-dev libcapstone-dev pkg-config
RUN apt update && apt install -y git make clang-6.0 gcc g++ ruby sudo pciutils gtkwave
RUN apt update && apt install -y build-essential dpkg-dev fakeroot kmod libalgorithm-diff-perl libalgorithm-diff-xs-perl libalgorithm-merge-perl libfakeroot libfile-fcntllock-perl liblocale-gettext-perl

# Install git-lfs
RUN wget https://github.com/git-lfs/git-lfs/releases/download/v2.4.2/git-lfs-linux-amd64-2.4.2.tar.gz && tar xzvf git-lfs-linux-amd64-2.4.2.tar.gz && cd git-lfs-2.4.2 && sudo bash ./install.sh

##################################################
# glog
##################################################
RUN apt update && apt install -y libgoogle-glog-dev

##################################################
# Python
##################################################
RUN apt update && apt install -y python3.8 python3.8-dev python3.8-venv python3-pip python3-setuptools python3-wheel
RUN python3 -m pip install python-gitlab elasticsearch torch

############################################
#  Rose
############################################
RUN apt update && apt install -y software-properties-common
RUN add-apt-repository ppa:rosecompiler/rose-development
RUN apt update && apt install -y rose rose-tools

############################################
# miniconda
############################################
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/miniconda
RUN rm -f miniconda.sh

############################################
#  Grayskull Riscv
############################################
RUN mkdir -p /opt
RUN curl "https://yyz-gitlab.local.tenstorrent.com/api/v4/projects/11/packages/generic/riscv32i/2020.11.13/riscv32i-v20201113.tar.gz" --output /opt/riscv32i-v20201113.tar.gz
RUN tar -xvzf /opt/riscv32i-v20201113.tar.gz -C /opt
RUN rm /opt/riscv32i-v20201113.tar.gz

# Extract seperate risc toolchain for Tensix team
RUN curl "https://yyz-gitlab.local.tenstorrent.com/api/v4/projects/11/packages/generic/riscv64iafv/2022.03.17/riscv64iafv.tar.gz" --output /opt/riscv64iafv.tar.gz
RUN tar -xvzf /opt/riscv64iafv.tar.gz -C /opt
RUN rm /opt/riscv64iafv.tar.gz

##################################################
# ccache
##################################################
RUN apt install -y ccache=3.7.7-1
#Across docker runners, we try to share mounted ccache dir
RUN mkdir -p /runner-ccache

##################################################
# Set up filesystem
##################################################
# Shared home directory and nextstep
RUN apt update && apt install -y nfs-common cifs-utils

##################################################
# Set up systemd
##################################################
RUN apt install -y --no-install-recommends software-properties-common rsyslog systemd systemd-cron sudo iproute2

##################################################
# Add new packages here to speed up cached builds - once all builds are versioned caching can be turned on
##################################################
# LibGL for YOLO
RUN apt install -y libgl1-mesa-glx
# Bazel
RUN apt install -y openjdk-8-jdk && echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && apt update && apt install -y bazel
# rsync
RUN apt install -y rsync
# bear
RUN apt install -y bear
# tqdm
RUN pip3 install tqdm
# gmock and gtest
RUN apt install -y libgtest-dev libgmock-dev
# perf
RUN apt update && \
    apt install -y linux-tools-generic linux-cloud-tools-generic linux-tools-common && \
    update-alternatives --install /usr/local/bin/perf perf "$(find /usr/lib/linux-tools*/perf | head -1)" 2 && \
    update-alternatives --install /usr/local/bin/perf perf /usr/bin/perf 1 && \
    setcap cap_sys_admin,cap_sys_ptrace,cap_syslog=ep "$(readlink -f $(which perf))"
# clang-format
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add - && \
    add-apt-repository 'deb http://apt.llvm.org/focal/ llvm-toolchain-focal main' && \
    apt update && \
    apt install -y clang-format
# llvm-17
RUN wget https://apt.llvm.org/llvm.sh -P /tmp \
    && chmod +x /tmp/llvm.sh && \
    /tmp/llvm.sh 17

FROM base-ubuntu-20-04-amd64 as pybuda-ubuntu-20-04-amd64
# Set up Gitlab runner
ARG LOGIN=tester
# Add user and grant sudo privileges. Password will be the same as login.
RUN useradd -m -s /bin/bash -G sudo tester
RUN echo "tester:tester" | chpasswd
# Copy a few common files into the image
RUN chown -R tester:tester /home/tester
# Change the user that runs the gitlab-runner service to gitlab-runner. This
# fixes the non-working cancel button in the Gitlab GUI. See the answer
# provided by Michael Gerber at https://gitlab.com/gitlab-org/gitlab-runner/issues/1662
RUN sed -i 's/--exec/--user gitlab-runner --exec/g' /etc/init.d/gitlab-runner
RUN sed -i 's/"--user" "gitlab-runner"//g' /etc/init.d/gitlab-runner
# The next one prevents the "mesg: ttyname failed: Inappropriate ioctl for device":
#   (see https://superuser.com/questions/1241548/xubuntu-16-04-ttyname-failed-inappropriate-ioctl-for-device#1253889)
RUN sed -i 's/mesg n/tty -s \&\& mesg n/g' /root/.profile
# Setup YYZ isilon home directory
RUN echo "yyz-isi-01-nfs:/ifs/data/home /home nfs rsize=8192,wsize=8192,timeo=14,intr" >> /etc/fstab
WORKDIR /home/tester
CMD ["/bin/bash"]