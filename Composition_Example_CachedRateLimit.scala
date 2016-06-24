// Find out if user is rate-limited. This can be slow; we  have to ask
// the remote server that keeps track of who is rate-limited.
def isRateLimited(u: User): Future[Boolean] = {
	// ...
}

// Notice how we can swap this implementation out now with something that
// might implement a different, more restrictive policy

// Check the cache to find out if user is rate-limited. This cache
// implementation is just a Map, and can return a value right away. But
// we return a Future anyhow in case we need to use a slower implementation
// later
def isLimitedByCache(u: User): Future[Boolean] = Future.value(limitCache(u))

// Updates the cache
def setIsLimitedInCache(user: User, v: Boolean) { limitCache(user) = v}

// Get a timeline of tweets... unless the user is rate-limited (then
// throw an exception instead)
def getTimeline(cred: Credentials): Future[Timeline] = 
	isLimitedByCache(Cred.user) flatMap {
		case true => Future.exception(new Exception("rate limited"))
		case false => {

			// First we get auth'd user then we get timeline.
			// Sequential composition of asynchronous APIs: use flatMap
			val timeline = auth(cred) flatMap(getTimeline)
			val limited = isRateLimited(cred.user) onSuccess(
													setIsLimitedInCache(cred.user, _))

			// 'join' concurrently combines differently-typed futures
			// 'flatMap' sequentially combines, specifies what to do next
			timeline join limited flatMap {
				case(_, true) => Future.exception(new Exception("rate limited"))
				case(timeline, _) => Future.value(timeline)
			}
		}
	}