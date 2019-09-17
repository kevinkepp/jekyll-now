---
layout: default
---

<div class="posts">
	{% assign about = site.pages | where: "title", "About" %}
	<article class="post">
		<h1>About</h1>

		<div class="entry">
        	{{ about }}
		</div>
	</article>

  	{% for post in site.posts %}
    	<article class="post">

			<h1><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h1>

			<div class="entry">
	        	{{ post.content }}
			</div>

			<!--<a href="{{ site.baseurl }}{{ post.url }}" class="read-more">Read More</a>-->
		</article>
	{% endfor %}
</div>
