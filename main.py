from src.style_transfer import StyleTransfer

if __name__ == "__main__":
    # content_path = 'images/Green_Sea_Turtle_grazing_seagrass.jpg'
    # style_path = 'images/The_Great_Wave_off_Kanagawa.jpg'
    # content_path = 'images/stephen_test_content.jpg'
    # style_path = 'images/stephen_test_style.jpg'
    # content_path = 'images/bobo_train.jpg'
    # content_path = 'images/nathan_pose.jpg'
    content_path = 'images/edited.jpg'
    # content_path = 'ebsynth/nathan_test/video/70.jpg'
    # content_path = 'images/nathan_owl.jpg'
    # style_path = 'images/ninja_cropped.jpg'
    # content_path = 'images/bobo_tea.jpg'
    # style_path = 'images/jojo.jpg'
    # style_path = 'images/zelda_shrine.jpg'
    style_path = 'images/hearts.jpg'
    st = StyleTransfer(content_path=content_path, style_path=style_path)
    st.run(num_iterations=1000)
