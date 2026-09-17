# ==============================================================================
# [실습] 지하철 노선망으로 배우는 탐색 알고리즘
# ==============================================================================

# [자료 1] 무가중치 그래프 (DFS / BFS 용)
subway_graph = {
    "시청": ["신도림", "동대문", "용산", "강남"],
    "신도림": ["시청", "동대문"],
    "동대문": ["시청", "신도림", "강남", "잠실"],
    "용산": ["시청", "강남"],
    "강남": ["시청", "동대문", "용산", "잠실"],
    "잠실": ["동대문", "강남"]
}

# [자료 2] 가중치 그래프 (다익스트라 용: 소요 시간(분))
subway_weighted_map = {
    "시청": [("신도림", 12), ("동대문", 10), ("용산", 8), ("강남", 15)],
    "신도림": [("시청", 12), ("동대문", 14)],
    "동대문": [("시청", 10), ("신도림", 14), ("강남", 13), ("잠실", 11)],
    "용산": [("시청", 8), ("강남", 9)],
    "강남": [("시청", 15), ("동대문", 13), ("용산", 9), ("잠실", 10)],
    "잠실": [("동대문", 11), ("강남", 10)]
}

subway_graph = {
    "시청": ["신도림", "동대문", "용산", "강남"],
    "신도림": ["시청", "동대문"],
    "동대문": ["시청", "신도림", "강남", "잠실"],
    "용산": ["시청", "강남"],
    "강남": ["시청", "동대문", "용산", "잠실"],
    "잠실": ["동대문", "강남"]
}
# --- 1. DFS (깊이 우선 탐색) ---
def dfs(graph, start_station):
    visited = []
    stack = [start_station]

    while stack:
        current = stack.pop()  # 맨 뒤(가장 최근) 데이터 꺼내기

        if current not in visited:
            visited.append(current)
            for neighbor in reversed(graph[current]):
                if neighbor not in visited:
                    stack.append(neighbor)
    return visited


# --- 2. BFS (너비 우선 탐색) ---
def bfs(graph, start_station):
    visited = []
    queue = [start_station]

    while queue:
        current = queue.pop(0)  # 줄의 맨 앞(가장 먼저 들어온 것) 데이터 꺼내기

        if current not in visited:
            visited.append(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
    return visited


# --- 3. 다익스트라 최단 경로 알고리즘 ---
def dijkstra(graph, start_station, end_station):
    times = {station: 999999 for station in graph}
    times[start_station] = 0
    previous = {station: None for station in graph}
    visited = []

    for _ in range(len(graph)):
        # 미방문 정류장 중 소요 시간이 가장 작은 곳 선택
        current_station = None
        min_time = 999999

        for station in graph:
            if station not in visited and times[station] < min_time:
                min_time = times[station]
                current_station = station

        if current_station is None:
            break

        visited.append(current_station)

        # 이웃 정류장 소요 시간 갱신
        for next_station, travel_time in graph[current_station]:
            new_time = times[current_station] + travel_time
            if new_time < times[next_station]:
                times[next_station] = new_time
                previous[next_station] = current_station

    # 경로 역추적
    path = []
    curr = end_station
    while curr is not None:
        path.append(curr)
        if curr == start_station:
            break
        curr = previous[curr]

    if path[-1] != start_station:
        return None, []

    return times[end_station], list(reversed(path))


# [자료 3] 역별 2D 지리 좌표 (A* 휴리스틱 계산용: 서울 실제 방위각 기반 좌표)
subway_coordinates = {
    "시청": (0.0, 0.35),
    "동대문": (0.65, 0.40),
    "신도림": (-0.85, -0.35),
    "용산": (-0.15, -0.10),
    "강남": (0.35, -0.55),
    "잠실": (0.85, -0.45)
}


# --- 4. A* (에이스타) 최단 경로 알고리즘 ---
def astar(graph, start_station, end_station, coordinates=None, heuristic_type="euclidean", weight=1.0):
    """
    A* 최단 경로 알고리즘:
    f(n) = g(n) + w * h(n)
    - g(n): 출발역에서 현재 역까지의 실제 소요 시간
    - h(n): 현재 역에서 도착역까지의 예상 잔여 시간 (휴리스틱 추정치)
    - f(n): 총 예상 소요 시간 (f값이 가장 작은 역을 우선 탐색)
    - weight: 휴리스틱 가중치 (w=0: 다익스트라, w=1: 표준 A*, w>1: 탐욕적 A*)
    """
    import math

    if coordinates is None:
        coordinates = subway_coordinates

    def get_heuristic(u, target):
        if heuristic_type == "zero" or u not in coordinates or target not in coordinates:
            return 0.0
        x1, y1 = coordinates[u]
        x2, y2 = coordinates[target]
        if heuristic_type == "manhattan":
            dist = abs(x1 - x2) + abs(y1 - y2)
        else:  # euclidean
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        # 축척 계수: 좌표 1단위 거리당 약 12분으로 환산 (최단 경로 보장용 Admissible 조건 충족)
        return round(dist * 12.0, 1)

    # g_scores: 시작역부터 각 역까지의 실제 소요 시간
    g_scores = {station: 999999 for station in graph}
    g_scores[start_station] = 0

    # f_scores: g(n) + w * h(n)
    f_scores = {station: 999999 for station in graph}
    f_scores[start_station] = round(weight * get_heuristic(start_station, end_station), 1)

    previous = {station: None for station in graph}
    open_set = [start_station]  # 대기 목록
    visited = []  # 확정(Closed) 목록

    while open_set:
        # open_set 중에서 f_score가 가장 작은 역 선택
        current = min(open_set, key=lambda stn: f_scores[stn])

        if current == end_station:
            visited.append(current)
            break

        open_set.remove(current)
        visited.append(current)

        for next_station, travel_time in graph.get(current, []):
            if next_station in visited:
                continue

            tentative_g = g_scores[current] + travel_time

            if tentative_g < g_scores[next_station]:
                g_scores[next_station] = tentative_g
                h_val = get_heuristic(next_station, end_station)
                f_scores[next_station] = round(tentative_g + weight * h_val, 1)
                previous[next_station] = current

                if next_station not in open_set:
                    open_set.append(next_station)

    # 경로 역추적
    path = []
    curr = end_station
    while curr is not None:
        path.append(curr)
        if curr == start_station:
            break
        curr = previous[curr]

    if not path or path[-1] != start_station:
        return None, []

    return g_scores[end_station], list(reversed(path))