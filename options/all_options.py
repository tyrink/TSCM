from .base_options import BaseOptions

class AllOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')       
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")        
        self.parser.add_argument("--Arc_path", type=str, default='arcface_model/arcface_checkpoint.tar', help="run ONNX model via TRT")
        self.parser.add_argument("--pic_a_path", type=str, default='./crop_224/gdg.jpg', help="Person who provides identity information")
        self.parser.add_argument("--pic_b_path", type=str, default='./crop_224/zrf.jpg', help="Person who provides information other than their identity")
        self.parser.add_argument("--pic_specific_path", type=str, default='./crop_224/zrf.jpg', help="The specific person to be swapped")
        self.parser.add_argument("--multisepcific_dir", type=str, default='./demo_file/multispecific', help="Dir for multi specific")
        self.parser.add_argument("--video_path", type=str, default='./demo_file/multi_people_1080p.mp4', help="path for the video to swap")
        self.parser.add_argument("--temp_path", type=str, default='./temp_results', help="path to save temporarily images")
        self.parser.add_argument("--output_path", type=str, default='./output/FF/', help="results path")
        self.parser.add_argument('--id_thres', type=float, default=0.03, help='how many test images to run')
        self.parser.add_argument('--no_simswaplogo', default=True, action='store_true', help='Remove the watermark')
        self.parser.add_argument('--use_mask', action='store_true', help='Use mask for better result')
        self.parser.add_argument('--dataroot', type=str, default='/data2/lfy/data/FFHQ/test/imgs')

        # self.parser.add_argument("--Arc_path", type=str, default='arcface_model/arcface_checkpoint.tar',
        #                          help="run ONNX model via TRT")
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=99, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=10000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        # self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        # self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=10000, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=10000, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
 
        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=4, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=20.0, help='weight for feature matching loss')
        self.parser.add_argument('--lambda_id', type=float, default=20.0, help='weight for id loss')
        self.parser.add_argument('--lambda_rec', type=float, default=20.0, help='weight for reconstruction loss')
        self.parser.add_argument('--lambda_GP', type=float, default=0.00001, help='weight for gradient penalty loss')
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        self.parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--times_G', type=int, default=1, help='time of training generator before traning discriminator')
        # self.parser.add_argument('--dataroot_FF', type=str, default='/data2/lfy/data/FFHQ')
        # self.parser.add_argument('--dataroot_VC', type=str, default='/data2/lfy/data/VoxCeleb2/cropped')
        # self.parser.add_argument('--dataroot_CA', type=str, default='/data2/lfy/data/CelebAHQ')
        # self.parser.add_argument('--dataroot_vgg', type=str, default='/data2/lfy/data/vggface2_crop_arcfacealign_224')


        
 
        self.isTrain = False
