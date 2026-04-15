#!/usr/bin/env python3
"""
Google Drive Batch Downloader
从gdrive_*.html文件中提取下载链接并批量下载
用法: python download_datasets.py --html_dir /path/to/html/files --output_dir /path/to/save
"""

import sys
import time
import argparse
import requests
from pathlib import Path
from bs4 import BeautifulSoup


def parse_gdrive_html(html_path):
    """从Google Drive virus warning页面HTML中提取下载信息"""
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'html.parser')
    form = soup.find('form', id='download-form')
    if not form:
        print(f"  [!] 找不到下载表单: {html_path}")
        return None

    # 提取所有hidden input
    params = {}
    for inp in form.find_all('input', type='hidden'):
        params[inp.get('name')] = inp.get('value')

    # 提取文件名
    name_tag = soup.find('a', href=lambda h: h and '/open?id=' in h)
    filename = name_tag.text.strip() if name_tag else Path(html_path).stem + '.zip'

    # 提取action URL
    action = form.get('action', 'https://drive.usercontent.google.com/download')

    return {
        'filename': filename,
        'action': action,
        'params': params,
        'html_path': str(html_path)
    }


def download_file(info, output_dir, chunk_size=1024 * 1024 * 8):
    """下载单个文件，支持断点续传"""
    output_path = Path(output_dir) / info['filename']

    # 检查是否已下载完成
    if output_path.exists():
        print(f"  [✓] 已存在，跳过: {info['filename']}")
        return True

    # 检查临时文件（断点续传）
    tmp_path = output_path.with_suffix(output_path.suffix + '.tmp')
    downloaded = tmp_path.stat().st_size if tmp_path.exists() else 0

    headers = {}
    if downloaded > 0:
        headers['Range'] = f'bytes={downloaded}-'
        print(f"  [→] 断点续传，已下载 {downloaded / 1024 / 1024:.1f} MB: {info['filename']}")
    else:
        print(f"  [↓] 开始下载: {info['filename']}")

    try:
        session = requests.Session()
        response = session.get(
            info['action'],
            params=info['params'],
            headers=headers,
            stream=True,
            timeout=60,
        )
        response.raise_for_status()

        # 获取总大小
        total = int(response.headers.get('content-length', 0)) + downloaded
        total_mb = total / 1024 / 1024

        mode = 'ab' if downloaded > 0 else 'wb'
        with open(tmp_path, mode) as f:
            current = downloaded
            last_print = time.time()
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    current += len(chunk)
                    # 每5秒打印一次进度
                    if time.time() - last_print > 5:
                        pct = current / total * 100 if total > 0 else 0
                        print(f"    {current / 1024 / 1024:.1f}/{total_mb:.1f} MB ({pct:.1f}%)")
                        last_print = time.time()

        # 下载完成，重命名
        tmp_path.rename(output_path)
        print(f"  [✓] 下载完成: {info['filename']} ({total_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  [✗] 下载失败: {info['filename']} — {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Google Drive Batch Downloader')
    parser.add_argument('--html_dir', type=str, required=True,
                        help='包含gdrive_*.html文件的目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='下载文件保存目录')
    parser.add_argument('--pattern', type=str, default='gdrive_*.html',
                        help='HTML文件匹配模式 (默认: gdrive_*.html)')
    args = parser.parse_args()

    html_dir = Path(args.html_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 找所有HTML文件
    html_files = sorted(html_dir.glob(args.pattern))
    if not html_files:
        print(f"[!] 在 {html_dir} 中找不到匹配 {args.pattern} 的文件")
        sys.exit(1)

    print(f"[*] 找到 {len(html_files)} 个下载任务\n")

    success, fail = 0, 0
    for idx, html_path in enumerate(html_files, start=1):
        print(f"[{idx}/{len(html_files)}] 处理: {html_path.name}")
        info = parse_gdrive_html(html_path)
        if info is None:
            fail += 1
            continue
        print(f"  文件名: {info['filename']}")
        ok = download_file(info, output_dir)
        if ok:
            success += 1
        else:
            fail += 1
        print()

    print(f"\n[完成] 成功: {success}  失败: {fail}")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Google Drive Batch Downloader
从gdrive_*.html文件中提取下载链接并批量下载
用法: python download_datasets.py --html_dir /path/to/html/files --output_dir /path/to/save
"""

import os
import re
import sys
import time
import argparse
import requests
from pathlib import Path
from bs4 import BeautifulSoup


def parse_gdrive_html(html_path):
    """从Google Drive virus warning页面HTML中提取下载信息"""
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'html.parser')
    form = soup.find('form', id='download-form')
    if not form:
        print(f"  [!] 找不到下载表单: {html_path}")
        return None

    # 提取所有hidden input
    params = {}
    for inp in form.find_all('input', type='hidden'):
        params[inp.get('name')] = inp.get('value')

    # 提取文件名
    name_tag = soup.find('a', href=lambda h: h and '/open?id=' in h)
    filename = name_tag.text.strip() if name_tag else Path(html_path).stem + '.zip'

    # 提取action URL
    action = form.get('action', 'https://drive.usercontent.google.com/download')

    return {
        'filename': filename,
        'action': action,
        'params': params,
        'html_path': str(html_path)
    }


def download_file(info, output_dir, chunk_size=1024*1024*8):
    """下载单个文件，支持断点续传"""
    output_path = Path(output_dir) / info['filename']

    # 检查是否已下载完成
    if output_path.exists():
        print(f"  [✓] 已存在，跳过: {info['filename']}")
        return True

    # 检查临时文件（断点续传）
    tmp_path = output_path.with_suffix(output_path.suffix + '.tmp')
    downloaded = tmp_path.stat().st_size if tmp_path.exists() else 0

    headers = {}
    if downloaded > 0:
        headers['Range'] = f'bytes={downloaded}-'
        print(f"  [→] 断点续传，已下载 {downloaded/1024/1024:.1f} MB: {info['filename']}")
    else:
        print(f"  [↓] 开始下载: {info['filename']}")

    try:
        session = requests.Session()
        response = session.get(
            info['action'],
            params=info['params'],
            headers=headers,
            stream=True,
            timeout=60
        )
        response.raise_for_status()

        # 获取总大小
        total = int(response.headers.get('content-length', 0)) + downloaded
        total_mb = total / 1024 / 1024

        mode = 'ab' if downloaded > 0 else 'wb'
        with open(tmp_path, mode) as f:
            current = downloaded
            last_print = time.time()
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    current += len(chunk)
                    # 每5秒打印一次进度
                    if time.time() - last_print > 5:
                        pct = current / total * 100 if total > 0 else 0
                        print(f"    {current/1024/1024:.1f}/{total_mb:.1f} MB ({pct:.1f}%)")
                        last_print = time.time()

        # 下载完成，重命名
        tmp_path.rename(output_path)
        print(f"  [✓] 下载完成: {info['filename']} ({total_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  [✗] 下载失败: {info['filename']} — {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Google Drive Batch Downloader')
    parser.add_argument('--html_dir', type=str, required=True,
                        help='包含gdrive_*.html文件的目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='下载文件保存目录')
    parser.add_argument('--pattern', type=str, default='gdrive_*.html',
                        help='HTML文件匹配模式 (默认: gdrive_*.html)')
    args = parser.parse_args()

    html_dir = Path(args.html_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 找所有HTML文件
    html_files = sorted(html_dir.glob(args.pattern))
    if not html_files:
        print(f"[!] 在 {html_dir} 中找不到匹配 {args.pattern} 的文件")
        sys.exit(1)

    print(f"[*] 找到 {len(html_files)} 个下载任务\n")

    success, fail = 0, 0
    for html_path in html_files:
        print(f"[{html_files.index(html_path)+1}/{len(html_files)}] 处理: {html_path.name}")
        info = parse_gdrive_html(html_path)
        if info is None:
            fail += 1
            continue
        print(f"  文件名: {info['filename']}")
        ok = download_file(info, output_dir)
        if ok:
            success += 1
        else:
            fail += 1
        print()

    print(f"\n[完成] 成功: {success}  失败: {fail}")


if __name__ == '__main__':
    main()