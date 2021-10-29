from blpapi import Name, Session, SessionOptions, Event
from pandas import DataFrame
from datetime import datetime

BAR_DATA = Name("barData")
BAR_TICK_DATA = Name("barTickData")
OPEN = Name("open")
HIGH = Name("high")
LOW = Name("low")
CLOSE = Name("close")
VOLUME = Name("volume")
NUM_EVENTS = Name("numEvents")
TIME = Name("time")

TICK_DATA = Name("tickData")
COND_CODE = Name("conditionCodes")
TICK_SIZE = Name("size")
TYPE = Name("type")
VALUE = Name("value")

RESPONSE_ERROR = Name("responseError")
SESSION_TERMINATED = Name("SessionTerminated")
CATEGORY = Name("category")
MESSAGE = Name("message")

res = []


def print_error_info(leading_str, error_info):
    print("%s%s (%s)" % (leading_str, error_info.getElementAsString(CATEGORY),
                         error_info.getElementAsString(MESSAGE)))


def process_message(msg):
    if msg.hasElement(BAR_DATA):
        data = msg.getElement(BAR_DATA).getElement(BAR_TICK_DATA)
        for bar in data.values():
            res.append((bar.getElementAsDatetime(TIME), bar.getElementAsFloat(OPEN), bar.getElementAsFloat(HIGH),
                        bar.getElementAsFloat(LOW), bar.getElementAsFloat(CLOSE), bar.getElementAsInteger(NUM_EVENTS),
                        bar.getElementAsInteger(VOLUME)))
    elif msg.hasElement(TICK_DATA):
        data = msg.getElement(TICK_DATA).getElement(TICK_DATA)
        for item in data.values():
            res.append((item.getElementAsDatetime(TIME), item.getElementAsString(TYPE), item.getElementAsFloat(VALUE),
                        item.getElementAsInteger(TICK_SIZE),
                        item.getElementAsString(COND_CODE) if item.hasElement(COND_CODE) else ""))


def process_response_event(event):
    for msg in event:
        if msg.hasElement(RESPONSE_ERROR):
            print_error_info("REQUEST FAILED: ", msg.getElement(RESPONSE_ERROR))
            continue
        print(msg)
        process_message(msg)


def send_intraday_bar_request(session, security, start_date, end_date, interval, event_type):
    ref_data_service = session.getService("//blp/refdata")
    request = ref_data_service.createRequest("IntradayBarRequest")

    request.set("security", security)
    request.set("eventType", event_type)
    request.set("interval", interval)
    request.set("startDateTime", start_date)
    request.set("endDateTime", end_date)
    request.set("gapFillInitialBar", True)
    print(request)
    session.sendRequest(request)


def event_loop(session):
    done = False
    while not done:
        event = session.nextEvent(500)
        if event.eventType() == Event.PARTIAL_RESPONSE:
            process_response_event(event)
        elif event.eventType() == Event.RESPONSE:
            process_response_event(event)
            done = True
        else:
            for msg in event:
                if event.eventType() == Event.SESSION_STATUS:
                    if msg.messageType() == SESSION_TERMINATED:
                        done = True


def get_intraday_bar_data(securities, start_datetime, end_datetime, interval, event_type):
    session = Session(SessionOptions())
    # Start a Session
    if not session.start():
        print("Failed to start session.")
        return None
    try:
        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return None

        send_intraday_bar_request(session, securities, start_datetime, end_datetime, interval, event_type)
        global res
        res = []
        # wait for events from session.
        event_loop(session)
        df = DataFrame(data=res, columns=['time', 'open', 'high', 'low', 'close', 'num_events', 'volume'])
    finally:
        # Stop the session
        session.stop()
    return df


def sendIntradayTickRequest(session, *args):
    security, events, start_dt, end_dt = args
    refDataService = session.getService("//blp/refdata")
    request = refDataService.createRequest("IntradayTickRequest")

    request.set("security", security)

    # Add fields to request
    eventTypes = request.getElement("eventTypes")
    for event in events:
        eventTypes.appendValue(event)

    request.set("startDateTime", start_dt)
    request.set("endDateTime", end_dt)
    request.set("includeConditionCodes", True)
    session.sendRequest(request)


def get_intraday_tick_data(*args):
    session = Session(SessionOptions())
    if not session.start():
        print("Failed to start session.")
        return None

    try:
        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return None

        sendIntradayTickRequest(session, *args)

        global res
        res = []
        event_loop(session)
        df = DataFrame(data=res, columns=['time', 'type', 'value', 'size', 'cc'])
    finally:
        # Stop the session
        session.stop()
    return df


if __name__ == "__main__":
    security = 'TPZ1 Index'
    start_dt, end_dt = datetime(2021, 10, 1), datetime(2021, 10, 2)
    try:
        # print("IntradayBarExample")
        # df = get_intraday_bar_data(security, start_dt, end_dt, 1, 'TRADE')
        # print(df)

        print("IntradayTickExample")
        df = get_intraday_tick_data(security, ['TRADE', 'BID', 'ASK'], start_dt, end_dt)
        print(df.head())

    except KeyboardInterrupt:
        print("Ctrl+C pressed. Stopping...")
