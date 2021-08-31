"""nebweb URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from rest_framework import routers

from videos.views import (
    video_annotate,
    processing_page, settings_page, home_page,
    VideoSearchView, VideoUploadView, VideoProcessingMonitorViewSet,
    VideosExportView, ExportVideoGraphAsJSONView, ExportVideoGraphAsSVGView,
    ExportVideosGraphAsMLFile, ShowSimilarView, VideoDetailPageView,
    VideoDetailMomentsView, VideoDetailRecommendationsView
)

router = routers.DefaultRouter()
router.register(r'vpms', VideoProcessingMonitorViewSet, basename='vpm')

urlpatterns = [
    path('', home_page, name='home-page'),
    path('', include(router.urls)),
    path('admin/', admin.site.urls),
    path('search/', VideoSearchView.as_view(), name='video-search'),
    path('show-similar/', ShowSimilarView.as_view(), name='fetch-similar-videos'),
    path('upload/', VideoUploadView.as_view(), name='video-upload'),
    path('search/annotate', video_annotate, name='video-search-annotate'),
    path('processing/', processing_page, name='processing-page'),
    path('settings/', settings_page, name='settings-page'),
    path('export/video/csv/', VideosExportView.as_view(), name='export-videos-csv'),
    path('export/video/graphdb/json/', ExportVideoGraphAsJSONView.as_view(), name='export-graph-db-json'),
    path('export/video/graphdb/png/', ExportVideoGraphAsSVGView.as_view(), name='export-graph-db-png'),
    path('export/videos/graphdb/gml/', ExportVideosGraphAsMLFile.as_view(), name='export-graph-db-gml'),
    path('video/', VideoDetailPageView.as_view(), name='video-detail-page'),
    path('video/find-video-moments/', VideoDetailMomentsView.as_view(), name='find-video-moments'),
    path('video/video-recommendations/', VideoDetailRecommendationsView.as_view(), name='get-video-recommendations')
]
urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
