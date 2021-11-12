# MHEGRU
GRU implemented with MHEAAN

## Install
~~~
mkdir build
cd build
cmake ..
make
~~~

## Figure 1

1. debug 모드로 실행해서 출력되는 로그를 num_threads_[num_threads].txt 로 기록

```bash
$ ./bulid/MHEGRU -d -n [num_threads] | tee path_to_log_files/num_threads_[num_threads].txt
```

![Untitled](https://user-images.githubusercontent.com/6984542/141432233-0dc3c44b-9566-4ca2-90c0-8332088d7ef3.png)

2. 파이썬 코드로 로그 파싱 (같은 operation 이 여러 번 실행되면 average를 취합니다)

```bash
$ python python/plots/parse_profile.py --input path_to_log_files
```

3. profile_kernel.csv 생성

![Untitled](https://user-images.githubusercontent.com/6984542/141432308-36ca3c7e-4bd5-453f-bbe3-65f3fbed3b18.png)
