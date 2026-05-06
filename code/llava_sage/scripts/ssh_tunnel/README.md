# SSH 反向 SOCKS 隧道：让电脑 B 走电脑 A 的外网

场景：

- 电脑 A 能访问外网
- 电脑 B 不能直接访问外网
- 电脑 A 可以 SSH 到电脑 B

这个目录提供了一套最小可用脚本，让 B 把出站流量通过 A 转发出去。

## 文件说明

- `start_a_to_b_reverse_socks.sh`：在电脑 A 上运行
- `proxy_env_on_b.sh`：在电脑 B 上 `source`，导出代理环境变量
- `test_proxy_on_b.sh`：在电脑 B 上运行，验证隧道是否连通

## 原理

A 端脚本会同时做两件事：

1. 用 `ssh -D` 在 A 本机打开一个 SOCKS5 代理
2. 用 `ssh -R` 把这个 SOCKS5 代理反向暴露到 B 的本地端口

之后，B 上的程序只需要连：

`socks5h://127.0.0.1:11080`

这里的 `socks5h` 里的 `h` 很重要，表示 DNS 解析也走 A，而不是让 B 自己解析。

## 当前环境里观察到的地址

我在这台 B 上看到当前 SSH 会话对应的是：

- B 地址：`172.17.0.10`
- A 地址：`10.10.1.1`

如果你的网络没变，可以先把 `172.17.0.10` 当作 `B_HOST` 使用。

## 第 1 步：把脚本拿到 A，并在 A 上启动隧道

如果你想直接从 A 把这套脚本拷过去，可以在 A 上执行：

```bash
scp -r root@172.17.0.10:/path/to/sage_repro_bundle/scripts/ssh_tunnel ~/ssh_tunnel
cd ~/ssh_tunnel
chmod +x *.sh
```

然后在 A 上启动：

```bash
B_HOST=172.17.0.10 B_USER=root ./start_a_to_b_reverse_socks.sh
```

这个进程需要持续保持运行，建议放在 `tmux` 或 `screen` 里。

## 第 2 步：在 B 上启用代理

在 B 的 shell 里执行：

```bash
source /path/to/sage_repro_bundle/scripts/ssh_tunnel/proxy_env_on_b.sh
curl -I https://example.com
```

如果你不想改整个 shell 的环境变量，也可以只给单条命令加代理：

```bash
curl --proxy socks5h://127.0.0.1:11080 -I https://example.com
```

## 第 3 步：在 B 上测试隧道

```bash
/path/to/sage_repro_bundle/scripts/ssh_tunnel/test_proxy_on_b.sh
```

成功时会看到类似输出：

```text
[ok] 127.0.0.1:11080 is listening
[ok] Reached https://example.com through socks5h://127.0.0.1:11080
[ok] Proxy egress IP: <A 的公网出口 IP>
```

## 备注

- 这台 B 的 `sshd` 当前是 `AllowTcpForwarding yes`，所以这个方案可用。
- 这台 B 的 `sshd` 当前是 `GatewayPorts no`，所以最稳妥的绑定方式是 `127.0.0.1`，不要直接暴露到公网或局域网。
- 如果隧道断开，A 端脚本会自动重连。

