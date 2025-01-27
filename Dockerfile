FROM nvcr.io/nvidia/pytorch:24.09-py3

# Accept build arguments for user/group IDs
ARG USER_ID
ARG GROUP_ID
ARG USER_NAME=userapp
ARG WORKSPACE
ARG CACHE

# Install system-wide packages
RUN pip install --upgrade pip
RUN pip install huggingface_hub plotly datasets nvidia-ml-py3
RUN pip install -U kaleido
RUN pip install stanford-stk megablocks deepspeed
RUN pip install nvidia-ml-py3
RUN pip install transformers==4.42.4


# Install tmux and copy config
RUN apt update && apt install tmux -y

# setup ssh
# Install OpenSSH server
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir -p /var/run/sshd && \
    echo "PermitRootLogin no" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config

# Create Triton directory
RUN mkdir -p /root/.triton/autotune

RUN groupadd -f -g ${GROUP_ID} ${USER_NAME} || true && \
    id -u ${USER_NAME} >/dev/null 2>&1 || useradd -l -u ${USER_ID} -g ${GROUP_ID} ${USER_NAME} && \
    install -d -m 0755 -o ${USER_NAME} -g ${USER_NAME} /home/${USER_NAME}

# Create directory for non-root pip installations
RUN mkdir -p /home/${USER_NAME}/.local && \
    chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/.local

RUN python3 -m pip install -v -U --no-build-isolation git+https://github.com/microsoft/tutel@c7559e8


# Set the root password for SSH (not recommended for production)
RUN echo "${USER_NAME}:sacs" | chpasswd

# Allow the user to connect via SSH
RUN echo "AllowUsers ${USER_NAME}" >> /etc/ssh/sshd_config

# Set environment variables
ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"
# ENV PIP_TARGET=/home/${USER_NAME}/.local

ENV WORKSPACE=${WORKSPACE}
ENV CACHE=${CACHE}

# Switch to non-root user
USER ${USER_NAME}

RUN echo "cd ${WORKSPACE}" >> /home/${USER_NAME}/.bashrc
# RUN echo "python3 start.py 1 1234 adnexus bookcorpus all" >> /home/${USER_NAME}/.bash_history
# RUN echo "export PATH=${WORKSPACE}/transformers/home/.local/bin:$PATH" >> /home/${USER_NAME}/.bashrc

# Expose SSH port
EXPOSE 22

# Start SSH daemon
CMD ["/usr/sbin/sshd", "-D"]