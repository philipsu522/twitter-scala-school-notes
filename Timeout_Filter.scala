class TimeoutFilter[Req, Rep] (
	timeout: Duration,
	exception: RequestTimeoutException,
	timer: Timer)
	extends Filter[Req, Rep, Req, Rep]
 {
 	def this(timeout: Duration, timer: Timer) =
 		this(timeout, new IndividualRequestTimeoutException(timeout), timer)

 	def apply(request: Req, service: Service[Req, Rep]): Future[Rep] = {
 		val res = service(request)

 		res.within(timer, timeout) rescue {
 			case _: java.util.concurrent.TimeoutException =>
 				res.cancel()
 				Trace.record(TimeoutFilter.TimeoutAnnotation)
 				Future.exception(exception)
 		}
 	}
 }