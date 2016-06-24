import com.twitter.util.{Try,Future,Promise}

// a fetchable thing
trait Resource {
	def imageLinks(): Seq[String]
	def links(): Seq[String]
}

// HTML pages can link to Imgs and to other HTML pages.
class HTMLPage(val i: Seq[String], val l: Seq[String]) extends Resource {
	def imageLinks() = i
	def links = l
}

// IMGs don't actually link to anything else
class Img() extends Resource {
	def imageLinks = Seq()
	def links = Seq()
}

// profile.html links to gallery.html and has an image link to portrait.jpg
val profile = new HTMLPage(Seq("portrait.jpg"), Seq("gallery.html"))
val portrait = new Img

// gallery.html links to profile.html and two images
val gallery = new HTMLPage(Seq("kitten.jpg", "puppy.jpg"), Seq("profile.html"))
val kitten = new Img
val puppy = new Img

val internet = Map(
	"profile.html" -> profile,
	"gallery.html" -> gallery,
	"portrait.jpg" -> portrait,
	"kitten.jpg" -> kitten,
	"puppy.jpg" -> puppy
)

// fetch(url) attempts to fetch a resource from our fake internet.
// Its returned Future might contain a Resource or an exception
def fetch(url: String) = { new Promise(Try(internet(url))) }




// Sequential Composition
// not using combinators
def getThumbnail(url: String): Future[Resource] = {
	val returnVal = new Promise[Resource]
	fetch(url) onSuccess { page => // callback for successful page fetch
		fetch(page.imageLinks()(0)) onSuccess { p => // callback for successuful image fetch
			returnVal.setVal(p)
		} onFailure { exc => // callback for failed img fetch
			returnVal.setException(exc)
		}
	} onFailure { exc => // callback for failed page fetch
		returnVal.setException(exc)
	}
	returnVal
}
// using combinators - want page then img (sequential)
def getThumbnail(url: String): Future[Resource] = {
	fetch(url) flatMap { page => fetch(page.imageLinks()(0)) }
}



// Concurrent Composition
// if we want to fetch all images, not just first
def getThumbnails(url: String): Future[Seq[Resource]] = {
	fetch(url) flatMap { page =>
		Future.collect(
			page.imageLinks map { u => fetch(u) } // This is collection's map, depends on the thing before the map
		)
	}
}



// Concurrent + Recursion
// Instead of fetching a page’s images, we might fetch the other pages that it links to. If we then recurse on those, we have a simple web crawler.
// Return
def crawl(url: String): Future[Seq[Resource]] = {
	fetch(url) flatMap { page =>
		Future.collect(
			page.links map {u => crawl(u) }
		) map { pps => pps.flatten }
	}
}
