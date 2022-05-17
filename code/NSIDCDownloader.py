import csv, glob, io, os, re, requests, boto3
from genericpath import exists
from botocore.errorfactory import ClientError
from ftplib import FTP

from dotenv import load_dotenv

load_dotenv()


class NSIDCDownloader:
    SERVER_URL = 'sidads.colorado.edu'

    def __init__(self, remote_directory, s3_info={}):
        '''
            Initialized with the remote directory that is being accessed
            Also, the s3 info if we're uploading to s3
        '''
        self._connect(NSIDCDownloader.SERVER_URL)
        self.remote_directory = remote_directory
        self.cache = {}
        self.s3_info = s3_info

        if self.s3_info:
            # Creating a session to s3
            session = boto3.Session(
                aws_access_key_id=os.getenv('AWS_SERVER_PUBLIC_KEY'),
                aws_secret_access_key=os.getenv('AWS_SERVER_SECRET_KEY'),
            )
            self.s3 = session.client('s3')

    def _connect(self, server):
        '''
            Connect to the FTP server
        '''
        self.ftp = FTP(server)
        self.ftp.login()

    def discover_from_cmr(self, cmr_params, upload=False):
        '''
            Discovers all the granules in the CMR based on the parameters
            Uploads the corresponding tif file to s3 (if True)
        '''

        base_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        granules = requests.post(base_url, data=cmr_params)
        granules = granules.json()["feed"]["entry"]

        # Filter out all granule (.tif) URLs
        urls = []
        for granule in granules:
            for link in granule["links"]:
                if link["rel"] == "http://esipfed.org/ns/fedsearch/1.1/data#" and ".tif" in (href := link["href"]):
                    urls.append(href)

                    # Upload to s3
                    if upload and self.s3_info:
                        filename = href.split("/")[-1]
                        bucket, key = self.s3_info.get("bucket"), f"{self.s3_info.get('path')}{filename}"
                        if not self._exists(bucket, key):
                            print(f"Uploading {key} to s3")
                            username = os.environ.get("EARTHDATA_USERNAME")
                            password = os.environ.get("EARTHDATA_PASSWORD")
                            with requests.Session() as session:
                                session.auth = (username, password)
                                request = session.request("get", href)
                                response = session.get(request.url, auth=(username, password), stream=True)
                                self.s3.upload_fileobj(response.raw, bucket, key)
                    break
        return urls

    def _exists(self, bucket, key):
        try:
            # check if it already exists
            self.s3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False
    
    def download_file(self, filename, path, pattern="", upload=False):
        '''
            From the FTP directory,
            search for a file corresponding to `filename` in the given path inside
            a folder that follows the name pattern `pattern`
            Also upload to s3 (if True)
        '''
        self.ftp.cwd(f"{self.remote_directory}{path}")
        # print(f"{self.remote_directory}{path}_{filename}")
        # If the folder has already been found before, use that, else determine
        if pattern not in self.cache:
            folders = self.ftp.nlst()
            folders = folders[2:]
            folder = [f for f in folders if re.match(pattern, f)][0]
        else:
            folder = self.cache(pattern)

        self.ftp.cwd(f"{folder}")
        files = self.ftp.nlst()
        filename_pattern = re.match(f"RDSISC4_(.*)_classified", filename).group(1)
        target_file = [file for file in files if re.match(filename_pattern, file)][0]

        if not upload or not self.s3_info:
            print(f"The file is: {target_file}")
            return files

        # Write to s3
        bucket, key = self.s3_info.get("bucket"), f"{self.s3_info.get('path')}{target_file}"
        if not self._exists(bucket, key):
            # if not, write
            print(f"Uploading {key} to s3")
            myfile = io.BytesIO()
            self.ftp.retrbinary(f"RETR {target_file}", myfile.write)
            myfile.seek(0)
            self.s3.put_object(Body=myfile, Bucket=bucket, Key=key)

        return files


def get_filenames_from_qa(qa_dir, quality_threshold, melt_pond_area):
    files = []
    csv_files = glob.glob(f"{qa_dir}/GR_*.csv")
    for csv_file in csv_files:
        with open(csv_file) as fp:
            reader = csv.reader(fp, delimiter=",")
            next(reader, None)  # skip the headers
            for row in reader:
                if float(row[1]) >= quality_threshold and float(row[4]) > melt_pond_area:
                    match = re.match(".*GR_(\d{4})(\d{2})(\d{2})_metadata.csv", csv_file)
                    filepattern = f"RDSISC4_{match.group(1)}_{match.group(2)}_{match.group(3)}_{row[0].zfill(5)}_classified.tif"
                    files.append(filepattern)
    return files

def get_ymdf(cmr_url):
    # RDSISC4_2018_04_16_08327_classified.tif
    filename = cmr_url.split('/')[-1].split(".tif")[0]
    datetime_search = re.match('RDSISC4_(\d{4})_(\d{2})_(\d{2})_(\d{5})_classified', filename)
    year, month, day, frame = datetime_search.group(1), datetime_search.group(2), datetime_search.group(3), datetime_search.group(4)
    return year, month, day, frame, filename


if __name__=="__main__":

    # FTP directory
    icebridge_directory = '/pub/DATASETS/ICEBRIDGE/IODMS0_DMSraw_v01/'

    # s3 bucket/path where we'll upload the data
    s3_info = {
        "bucket": "veda-ai-supraglacial-meltponds",
        "path": "original/"
    }

    filenames = get_filenames_from_qa(os.path.abspath("../data/qa_files/"), quality_threshold=0.05, melt_pond_area=10000)
    print(f"Discovered {len(filenames)} from QA files.")
    
    # Params to discover granuels from the CMR
    cmr_params = {
        "short_name": "RDSISC4",
        "version": 1,
        "readable_granule_name": filenames[3999:4002],
        "page_size": 1000
    }

    # Downloader object
    downloader = NSIDCDownloader(icebridge_directory, s3_info)

    # Discover from CMR
    urls = downloader.discover_from_cmr(cmr_params, upload=True)
    print(f"Discovered {len(urls)} granules from the CMR.")

    for url in urls:
        try:
            year, month, day, frame, filename = get_ymdf(url)
            root_folder = f'{year}_GR_NASA/'
            downloader.download_file(
                filename,
                root_folder,
                pattern=f"{month}{day}{year}.*",
                upload=True
            )
        except Exception as e:
            print(f"Failed for {url.split('/')[-1]} index {urls.index(url)}")
            print(e)
            exit()
