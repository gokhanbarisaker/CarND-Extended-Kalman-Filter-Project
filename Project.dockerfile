FROM carnd-extended-kalman-filter-base

COPY . /usr/src/ekf
WORKDIR /usr/src/ekf
WORKDIR /usr/src/ekf/build

RUN cmake -DCMAKE_BUILD_TYPE=Debug ..
RUN make

CMD ["./ExtendedKF"]

# docker run -p 4567:4567 --rm -it carnd-extended-kalman-filter-project:latest