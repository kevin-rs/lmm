FROM rust:alpine AS builder

LABEL maintainer="Mahmoud Harmouch <oss@wiseai.dev>"
# hadolint ignore=DL3018
RUN apk update && apk upgrade && \
    apk add --no-cache \
    build-base \
    pkgconfig \
    openssl-dev \
    openssl-libs-static \
    git

WORKDIR /lmm
COPY Cargo.toml Cargo.lock README.md ./
COPY ./RUST.md ./
COPY ./WASM.md ./
COPY ./CLI.md ./
COPY ./AGENT.md ./
COPY ./DERIVE.md ./
COPY lmm ./lmm
COPY lmm-agent ./lmm-agent
COPY lmm-derive ./lmm-derive

RUN cargo build --release --features="rust-binary,net"

FROM alpine:3.22.4

# hadolint ignore=DL3018
RUN apk add --no-cache openssl ca-certificates sudo && \
    addgroup -S lmm && \
    adduser -S -G lmm lmm

RUN addgroup -S sudo && \
    addgroup lmm sudo && \
    echo "%sudo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /home/lmm

COPY --from=builder /lmm/target/release/lmm /usr/local/bin/lmm

USER lmm

ENTRYPOINT [ "/usr/local/bin/lmm" ]
