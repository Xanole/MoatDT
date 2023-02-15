import os
import sys
import time
import dpkt
import socket
import numpy as np
import math
from collections import Counter

UPSTREAM = 1
BOTH = 0
DOWNSTREAM = -1

def LocalIP(ip):
    if ip[0:3] == "10." or ip[0:4] == "172." or ip[0:4] == "192.":
        return True
    else:
        return False

def packet_time(pcap_path, direction):
    """
    print TCP payload length in directions of up (U), down (D) and both (b), respectively.
    :param str pcap_path: the pcap file's path
    :param str directon: the direction label
    :return list packet captured time sequence
    """
    packet_count = 0
    PACKET_SUM = 150
    start_time = 0.0

    time_sequence = []

    f = open(pcap_path, 'rb')
    pcap = dpkt.pcap.Reader(f)
    for ts, buf in pcap:
        time = ts

        if start_time == 0:
            start_time = time

        time_sequence.append(time - start_time)

        packet_count += 1
        if packet_count >= PACKET_SUM:
            break

    # print(time_sequence)
    return time_sequence


def entropy(s):
    """
    calcuate the entropy of a sequence
    :param str s: list
    """
    p, lns = Counter(s), float(len(s))
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())


def packet_size(pcap_path, direction):
    """
    print TCP payload length in directions of up (U), down (D) and both (b), respectively.
    :param str pcap_path: the pcap file's path
    :param str directon: the direction label
    :return list packet size statistic
    :return list entropy sequence
    """
    packet_count = 0
    PACKET_SUM = 150

    size_sequence = []
    entropy_sequence = []
    size_statistic = [0]*1461

    f = open(pcap_path, 'rb')
    pcap = dpkt.pcap.Reader(f)
    for ts, buf in pcap:
        time = ts

        eth = dpkt.ethernet.Ethernet(buf)
        ip = eth.data
        tcp = ip.data

        if hasattr(ip, 'src') and hasattr(ip, 'dst'):
            try:
                sip = socket.inet_ntop(socket.AF_INET, ip.src)
                dip = socket.inet_ntop(socket.AF_INET, ip.dst)
            except Exception as e:
                sip = socket.inet_ntop(socket.AF_INET6, ip.src)
                dip = socket.inet_ntop(socket.AF_INET6, ip.dst)

        p_direction = UPSTREAM if (LocalIP(sip)) else DOWNSTREAM
        length = len(tcp.data)

        size_sequence.append(length)
        entropy_sequence.append(entropy(size_sequence))
        size_statistic[length] += 1

        # if p_direction == direction:
        #     print(length)
        # elif p_direction == direction:
        #     print(-1 * length)
        # else:
        #     print(length)

        packet_count += 1
        if packet_count >= PACKET_SUM:
            break

    return size_statistic, entropy_sequence


def packet_direction(pcap_path, direction):
    """
    print TCP payload length in directions of up (U), down (D) and both (b), respectively.
    :param str pcap_path: the pcap file's path
    :param str directon: the direction label
    :return list ratio sequence down/up
    """
    packet_count = 0
    PACKET_SUM = 150

    up_packets = 0
    down_packets = 0
    ratio_sequence = []

    f = open(pcap_path, 'rb')
    pcap = dpkt.pcap.Reader(f)
    for ts, buf in pcap:
        time = ts

        eth = dpkt.ethernet.Ethernet(buf)
        ip = eth.data
        tcp = ip.data

        if hasattr(ip, 'src') and hasattr(ip, 'dst'):
            try:
                sip = socket.inet_ntop(socket.AF_INET, ip.src)
                dip = socket.inet_ntop(socket.AF_INET, ip.dst)
            except Exception as e:
                sip = socket.inet_ntop(socket.AF_INET6, ip.src)
                dip = socket.inet_ntop(socket.AF_INET6, ip.dst)

        p_direction = UPSTREAM if (LocalIP(sip)) else DOWNSTREAM
        length = len(tcp.data)

        if p_direction == UPSTREAM:
            up_packets += 1
            # print(length)
        elif p_direction == DOWNSTREAM:
            down_packets += 1
            # print(-1 * length)
        else:
            print(length)

        ratio_sequence.append(down_packets/up_packets)

        packet_count += 1
        if packet_count >= PACKET_SUM:
            break

    return ratio_sequence


if __name__ == '__main__':

    pcap_path_1 = '.\\Moat.pcap'
    pcap_path_2 = '.\\webpage.pcap'
    pcap_path_3 = '.\\video.pcap'
    pcap_path_4 = '.\\audio.pcap'
    pcap_path_5 = '.\\image.pcap'
    pcap_path_6 = '.\\file.pcap'

    # time-related comparison
    time_sequence_1 = packet_time(pcap_path_1, BOTH)
    time_sequence_2 = packet_time(pcap_path_2, BOTH)
    time_sequence_3 = packet_time(pcap_path_3, BOTH)
    time_sequence_4 = packet_time(pcap_path_4, BOTH)
    time_sequence_5 = packet_time(pcap_path_5, BOTH)
    time_sequence_6 = packet_time(pcap_path_6, BOTH)
    for i in range(150):
        print(i, time_sequence_1[i], time_sequence_2[i], time_sequence_3[i], time_sequence_4[i], time_sequence_5[i], time_sequence_6[i])

    # size-related comparison
    # size_statistic_1, entropy_sequence_1 = packet_size(pcap_path_1, BOTH)
    # size_statistic_2, entropy_sequence_2 = packet_size(pcap_path_2, BOTH)
    # size_statistic_3, entropy_sequence_3 = packet_size(pcap_path_3, BOTH)
    # size_statistic_4, entropy_sequence_4 = packet_size(pcap_path_4, BOTH)
    # size_statistic_5, entropy_sequence_5 = packet_size(pcap_path_5, BOTH)
    # size_statistic_6, entropy_sequence_6 = packet_size(pcap_path_6, BOTH)
    # # for i in range(1461):
    # #     print(i, size_statistic_1[i], size_statistic_2[i])
    # for i in range(150):
    #     print(i, entropy_sequence_1[i], entropy_sequence_2[i], entropy_sequence_3[i], entropy_sequence_4[i], entropy_sequence_5[i], entropy_sequence_6[i])

    # direction-related comparsion
    # ratio_sequence_1 = packet_direction(pcap_path_1, BOTH)
    # ratio_sequence_2 = packet_direction(pcap_path_2, BOTH)
    # ratio_sequence_3 = packet_direction(pcap_path_3, BOTH)
    # ratio_sequence_4 = packet_direction(pcap_path_4, BOTH)
    # ratio_sequence_5 = packet_direction(pcap_path_5, BOTH)
    # ratio_sequence_6 = packet_direction(pcap_path_6, BOTH)
    # for i in range(150):
    #     print(i, ratio_sequence_1[i], ratio_sequence_2[i], ratio_sequence_3[i], ratio_sequence_4[i], ratio_sequence_5[i], ratio_sequence_6[i])



